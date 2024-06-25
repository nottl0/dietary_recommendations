import random 
import os 
from ultralytics import YOLO
from utils import *
from argparse import ArgumentParser
import json
import yaml

parser = ArgumentParser()

# Get the patient case number
parser.add_argument("-c", "--number", dest="patient_case_number",
                    help="provide case number", metavar="INTEGER")

args = parser.parse_args()

img_dir = "/workspace/data/CAFSD/CAFSD/CAFSD/test/images/"
names_file = "/workspace/data/CAFSD/CAFSD/data.yaml"
meals_file = "/workspace/data/meal_types.json"
pretrained_model = "/workspace/model/yolov8x.pt"
model = YOLO(pretrained_model)

meal_dict = {}
with open(meals_file, "r") as f:
    meal_dict = json.load(f)

names = []
with open(names_file, "r") as f:
    names = yaml.safe_load(f)["names"]

all_files = [f for f in os.listdir(img_dir)]
types_found = []

while len(types_found) != 3:
    file = random.sample(all_files, 1)[0]
    print(file)
    full_file_path = os.path.join(img_dir, file)
    result = model(full_file_path, iou=0.5,  device=[0])[0]
    food_names = [names[int(index)] for index in result.boxes.cls]
    meal_t = meal_type(food_names, meal_dict)
    if meal_t != "" and meal_t not in types_found:
        print(f"{file} \n")
        print(meal_t)
        types_found.append(meal_t)
        os.system(f'python main.py -p {full_file_path} -c {args.patient_case_number}')
