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
