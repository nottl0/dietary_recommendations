# Function for looking up indices from the description_file
def get_food_indices(file_path):
    indices = []

    with open(file_path, 'r') as file:
        for line in file:
            nums = line.split(" ")

            if nums:
                try:
                    indices.append(int(nums[0]))

                except ValueError:
                    continue

    return indices


# Searching for the food classes respective to the indices
def find_names_by_indices(indices, names_list):
    present_names = []

    for index in indices:
        try:
            present_names.append(names_list[index])

        except IndexError:
            print("Index " + index + " is out of range for the names list.")

    present_names = list(set(present_names))

    return present_names


# Retrieving food information from the file of all food
def find_information_by_names(file_path, names):
    information = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for name in names:
            found = False
            xname = name + '-'

            for line in lines:
                if xname in line:
                    xname = '(' + xname + ')'
                    nline = ': ' + line.replace(xname, '')
                    information.append(nline)
                    found = True
                    break

            if not found:
                information.append(" ")

    text = ""
    for i in range(len(information)):
        text += names[i] + information[i] + "\n"
    return text


# Yielding case information from the case number
def find_patient_case_by_number(patients_file, case_num):
    case_description = ""

    with open(patients_file, 'r') as file:
        for line in file:
            l = "Case " + case_num
            if l in line:
                case_description += line

    if not case_description:
        raise ValueError("No case found for: ", case_num)
    
    return case_description


# Determining the meal type based on maximum number of class appearance
def meal_type(foods, meals):
    count_b, count_l, count_d = 0,0,0
    meal_type = ''
    for food in foods:
        if food in meals['breakfast']:
            count_b += 1
        if food in meals['lunch']:
            count_l += 1
        if food in meals['dinner']:
            count_d += 1

    if count_b == count_l == count_d == 0:
        meal_type = ''

    elif max(count_b, count_l, count_d) == count_b:
        meal_type = ' for breakfast'

    elif max(count_b, count_l, count_d) == count_d:
        meal_type = ' for dinner'

    else: 
        meal_type = ' for lunch'

    return meal_type


# Keep the bounding box predictions with the highest confidence
# Not used for now, but will be helpful when bboxes are needed
# THIS CAN SUBSTITUTE TO THE FIND_NAME_BY_INDEX FUNCTION 
# def filter_detected_items(result, names_list):
#     classes = []
#     bboxes = []
#     for food_index in set(result.boxes.cls):
#         if len(result.boxes.cls) < len(result.boxes.xywhn):
#             confs = [r.boxes.conf for r in result if r.boxes.cls == food_index]
#             bboxs = [r.boxes.xywhn for r in result if r.boxes.cls == food_index]
#             bbox_for_class = bboxs[confs.index(max(confs))]
#             try:
#                 classes.append(names_list[int(food_index)])

#             except IndexError:
#                 print("Index " + index + " is out of range for the names list.")
#         else:
#             bbox_for_class = result.boxes.xywhn
#         bboxes.append(bbox_for_class)
#     return classes, boxes

# Count number of food items to classify as meal
def is_meal(food_classes):
    return  3 <= len(food_classes) and len(food_classes) < 10


