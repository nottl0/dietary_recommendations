import json
import yaml
import os
import openai
from argparse import ArgumentParser
from ultralytics import YOLO
from utils import get_food_indices, find_names_by_indices, \
                find_information_by_names, find_patient_case_by_number, \
                is_meal, meal_type



# Get the image path
parser = ArgumentParser()
parser.add_argument("-p", "--path", dest="image_path",
                    help="provide path to image", metavar="IMAGE")

# Get the patient case number
parser.add_argument("-c", "--number", dest="patient_case_number",
                    help="provide case number", metavar="INTEGER")

args = parser.parse_args()

########################### YOLO INFERENCE ##############################

# Model inference
pretrained_model = "/workspace/model/yolov8x.pt"
model = YOLO(pretrained_model)

# Folder to store predictions
pred_dir = "/workspace/predictions"

# Yielding a prediction from a pre-trained model
result = model(args.image_path, iou=0.5)[0]

# Recording the model predictions for the given image in a file
if len(result.boxes.cls) != 0:

    decription_file = os.path.join(pred_dir, \
                                    os.path.splitext(os.path.basename(args.image_path))[0] + '.txt')

    with open(decription_file, 'a+') as f:
        for cls in result.boxes.cls:
            print(cls)
            f.write(str(int(cls.item())) + "\n")
        print(f.read())

        for x in result.boxes.xywhn[0]:
            f.write('')
            f.write(str(x.item()))

        f.write("\n")
        print(f.read())

else:
    raise ValueError("Could not Classify the Food Item.")

############################# GPT-API ####################################

# Api key
client = openai.OpenAI(
    api_key=""
)

# Using the model's predictions to convert to prompt argument
integer_file_path = decription_file

# A text file containing all descriptions of selected Central Asian foods
info_file_path = "/workspace/data/_modded_All_Food.txt"

# A text file containing all patient cases
patients_new = "/workspace/data/new_patients.txt"

# File with meal classes with corresponding food items
meals_file = "/workspace/data/meal_types.json"

# File with meal classes
names_file = "/workspace/data/CAFSD/CAFSD/data.yaml"

# List of food classes
names = []
with open(names_file, "r") as f:
    names = yaml.safe_load(f)["names"]

# Food classes according meal types 
meals_dict = {}
with open(meals_file, "r") as f:
    meals_dict = json.load(f)

# Getting the arguments for the GPT prompt
try:
    # Retrieving the indices from predictions
    indices = get_food_indices(integer_file_path)
    print("Found indices: ", indices)

    # Converting the indices into food classes from classes list
    food_present = find_names_by_indices(indices, names)
    print("Names found: ", food_present)

    # If there are less than 3 and more than 10 food items on
    # a given image, it is not considered to be particular meal
    meal = ""
    if is_meal(food_present) == True:
        
        # Defining the meal type for the prompt
        meal = meal_type(food_present, meals_dict)
        print("Meal type: ", meal)

    # Yielding information of the present classes
    food_desc = find_information_by_names(info_file_path, food_present)
    print("Information found: ", food_desc)

    # Getting the case information
    case_number = args.patient_case_number
    health_cond = find_patient_case_by_number(patients_new, case_number)
    print("Case found: ", health_cond)

except Exception as e:
    print("An error occurred: {e}")

user_prompt = "I need dietary recommendations on consuming the following food" + meal + ":\n" + food_desc + "The recommendations should be given to the patient with the following profile: " + health_cond
print(user_prompt)
with open(os.path.join("/workspace/responses/", \
            os.path.splitext(os.path.basename(args.image_path))[0] + '_' + args.patient_case_number + '_' + meal +'_prompt.txt'), "a+") as re:
    re.write(user_prompt)

params = {
    "food_present": food_present,
    "food_desc": food_desc,
    "health_cond": health_cond
}

completion = client.chat.completions.create(
    model="gpt-4",
    temperature=0.1,
    n=1,
    messages=[
        {"role": "user", "content": user_prompt},
    ],
)

response = completion.choices[0].message.content
print(response)

with open(os.path.join("/workspace/responses/", \
            os.path.splitext(os.path.basename(args.image_path))[0] + '_' + args.patient_case_number + '_' + meal + '.txt'), "a+") as re:
    re.write(response)
