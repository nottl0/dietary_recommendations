import json
import os
import openai
from argparse import ArgumentParser
from ultralytics import YOLO
from utils import get_food_indices, find_names_by_indices, \
                find_information_by_names, find_patient_case_by_number


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
pretrained_model = "/workspace/model/best.onnx"
model = YOLO(pretrained_model)

# Folder to store predictions
pred_dir = "/workspace/predictions"

# Yielding a prediction from a pre-trained model
results = model(args.image_path)

# Recording the model predictions for the given image in a file
for result in results:
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
    api_key="sk-proj-7KW75o0LYu9CctCftsefT3BlbkFJhqgfcZG409yNCtR501t0"
)

# Using the model's predictions to convert to prompt argument
integer_file_path = decription_file

# A text file containing all descriptions of selected Central Asian foods
info_file_path = "/workspace/data/_modded_All_Food.txt"

# A text file containing all patient cases
patients_new = "/workspace/data/new_patients.txt"

# List of classes
names = ['achichuk', 'airan-katyk', 'almond', 'apple', 'apricot', 'artichoke', 'arugula', 'asip', 'asparagus',
         'avocado', 'bacon', 'baklava', 'banana', 'basil', 'bauyrsak', 'bean soup', 'beans', 'beef shashlyk',
         'beef shashlyk-v', 'beer', 'beet', 'bell pepper', 'beshbarmak', 'beverages', 'black olives', 'blackberry',
         'blueberry', 'boiled chicken', 'boiled eggs', 'boiled meat', 'borsch', 'bread', 'brizol', 'broccoli',
         'buckwheat', 'butter', 'cabbage', 'cakes', 'carrot', 'cashew', 'casserole with meat and vegetables',
         'cauliflower', 'caviar', 'celery', 'cereal based cooked food', 'chak-chak', 'cheburek', 'cheese',
         'cheese souce', 'cherry', 'chestnuts', 'chicken shashlyk', 'chicken shashlyk-v', 'chickpeas', 'chili pepper',
         'chips', 'chocolate', 'chocolate paste', 'cinnabons', 'coffee', 'condensed milk', 'cooked eggplant',
         'cooked food based on meat', 'cooked food meat with vegetables', 'cooked tomatoes', 'cooked zucchini',
         'cookies', 'corn', 'corn flakes', 'crepe', 'crepe w filling', 'croissant', 'croissant sandwich', 'cucumber',
         'cutlet', 'dates', 'desserts', 'dill', 'doner-lavash', 'doner-nan', 'dragon fruit', 'dried fruits',
         'egg product', 'eggplant', 'figs', 'fish', 'french fries', 'fried cheese', 'fried chicken', 'fried eggs',
         'fried fish', 'fried meat', 'fruits', 'garlic', 'granola', 'grapefruit', 'grapes', 'green beans',
         'green olives', 'hachapuri', 'hamburger', 'hazelnut', 'herbs', 'hinkali', 'honey', 'hot dog', 'hummus',
         'hvorost', 'ice-cream', 'irimshik', 'jam', 'juice', 'karta', 'kattama-nan', 'kazy-karta', 'ketchup', 'kiwi',
         'kurt', 'kuyrdak', 'kymyz-kymyran', 'lagman-fried', 'lagman-w-soup', 'lavash', 'legumes', 'lemon', 'lime',
         'mandarin', 'mango', 'manty', 'mashed potato', 'mayonnaise', 'meat based soup', 'meat product', 'melon',
         'milk', 'minced meat shashlyk', 'mint', 'mixed berries', 'mixed nuts', 'muffin', 'mushrooms', 'naryn',
         'nauryz-kozhe', 'noodles soup', 'nuggets', 'oil', 'okra', 'okroshka', 'olivie', 'onion', 'onion rings',
         'orama', 'orange', 'pancakes', 'parsley', 'pasta', 'pastry', 'peach', 'peanut', 'pear', 'peas', 'pecan',
         'persimmon', 'pickled cabbage', 'pickled cucumber', 'pickled ginger', 'pickled squash', 'pie', 'pineapple',
         'pistachio', 'pizza', 'plov', 'plum', 'pomegranate', 'porridge', 'potatoes', 'pumpkin', 'pumpkin seeds',
         'quince', 'radish', 'raspberry', 'redcurrant', 'ribs', 'rice', 'rosemary', 'salad fresh', 'salad leaves',
         'salad with fried meat veggie', 'salad with sauce', 'samsa', 'sandwich', 'sausages', 'scallion', 'seafood',
         'seafood soup', 'sheep-head', 'shelpek', 'shorpa', 'shorpa chicken', 'smetana', 'smoked fish', 'snacks',
         'snacks bread', 'soda', 'souces', 'soup-plain', 'soy souce', 'spinach', 'spirits', 'strawberry', 'sugar',
         'sushi', 'sushi fish', 'sushi nori', 'sushki', 'suzbe', 'sweets', 'syrniki', 'taba-nan', 'talkan-zhent',
         'tartar', 'tea', 'tomato', 'tomato souce', 'tomato-cucumber-salad', 'tushpara-fried', 'tushpara-w-soup',
         'tushpara-wo-soup', 'vareniki', 'vegetable based cooked food', 'vegetable soup', 'waffles', 'walnut', 'wasabi',
         'water', 'watermelon', 'wine', 'wings', 'zucchini']

# Getting the arguments for the GPT prompt
try:
    # Retrieving the indices from predictions
    indices = get_food_indices(integer_file_path)
    print("Found indices: ", indices)

    # Converting the indices into food classes from classes list
    food_present = find_names_by_indices(indices, names)
    print("Names found: ", food_present)

    # Yielding information of the present classes
    food_desc = find_information_by_names(info_file_path, food_present)
    print("Information found: ", food_desc)

    # Getting the case information
    case_number = args.patient_case_number
    health_cond = find_patient_case_by_number(patients_new, case_number)
    print("Case found: ", health_cond)

except Exception as e:
    print("An error occurred: {e}")
        
user_prompt = "I need dietary recommendations on consuming the following food:\nFood items:\n" + food_desc + "The recommendations should be given to the patient with the following profile: " + health_cond
print(user_prompt)

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
            os.path.splitext(os.path.basename(args.image_path))[0] + '.txt'), "a+") as re:
    re.write("\n i)")
    re.write(response)
