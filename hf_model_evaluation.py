from tqdm import tqdm
import os
import json
import re
import ast

def list_files_sorted_by_size(folder_path):
    # Create a list of all files in the folder with their full paths
    files_with_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                        os.path.isfile(os.path.join(folder_path, file))]

    # Create a list of tuples (file size, file name)
    files_with_sizes = [(os.path.getsize(file), os.path.basename(file)) for file in files_with_paths]

    # Sort the list of tuples by file size
    files_sorted_by_size = sorted(files_with_sizes, key=lambda x: x[0])

    # Extract the file names, sorted by size
    sorted_file_names = [file_name for _, file_name in files_sorted_by_size]

    return sorted_file_names

def contains_non_numeric(matrix):
    return any(not isinstance(element, int) for element in matrix)

def parse_matrix(matrix_str):
    # Strip the outer brackets and split by commas, then trim whitespace
    elements = matrix_str.strip('[]').split(',')
    parsed_matrix = []
    for element in elements:
        try:
            # Try to convert each element to an integer
            parsed_matrix.append(int(element.strip()))
        except ValueError:
            # If it's not an integer, keep it as a string
            parsed_matrix.append(element.strip().strip('"').strip("'"))
    return parsed_matrix

def is_numeric_matrix(matrix):
    """Check if all elements in the matrix are numeric."""
    return all(isinstance(item, (int, float)) for sublist in matrix for item in sublist)


def extract_and_reconstruct(input_data):
    # Extract the output grid string from the input structure
    output_grid_string = input_data.split('</s>')[0]  # Assuming input_data is a string containing the output grid

    # Call the existing function to reconstruct the list of lists from this string
    reconstructed_data = reconstruct_from_string(output_grid_string)
    return reconstructed_data


def reconstruct_from_string(input_string):
    """
    Reconstructs the list of lists from the formatted string.
    """
    word_to_number = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }

    def process_line(line):
        # Extract the counts and numbers from the line
        parts = line.split('"')
        sublist = []
        for i in range(1, len(parts), 2):  # Only the parts that represent numbers
            count_word = parts[i - 1].strip().split()[-1]
            count = word_to_number.get(count_word, 1)  # Default to 1 if word not found
            number = int(parts[i].strip())
            sublist.extend([number] * count)
        return sublist

    # Split the input string into lines and process each one
    lines = input_string.strip().split('\n')
    data = [process_line(line) for line in lines if line.startswith('Row')]

    return data

def convert_language_to_matrix(input_string):
    dimension_pattern = r"The matrix dimensions are (\d+) columns by (\d+) rows"
    dimension_info = re.search(dimension_pattern, input_string)
    if dimension_info is None:
        raise ValueError("Could not find matrix dimensions in the input string.")
    cols = int(dimension_info.group(1))
    rows = int(dimension_info.group(2))

    # Extract coordinates and values
    coordinates_pattern = r"The coordinates of the non-zero elements, .*: \[(.*?)\]"
    coordinates_info = re.search(coordinates_pattern, input_string)
    if coordinates_info is None:
        raise ValueError("Could not find coordinates of non-zero elements in the input string.")
    coordinates_list = coordinates_info.group(1)
    coordinates = eval(f"[{coordinates_list}]")

    # Initialize the matrix with zeros
    matrix = [[0 for _ in range(cols)] for _ in range(rows)]

    # Populate the matrix with specified non-zero values
    for value, (x, y) in coordinates:
        matrix_index_y = (rows - 1) - y  # Adjust y-coordinate for the matrix (origin at bottom-left)
        matrix[matrix_index_y][x] = value  # Use the specified non-zero value
        # Optionally, print the matrix to verify
    return matrix


def string_to_matrix(matrix_string):
    """Converts a string representation of a matrix to a list of lists (matrix)."""
    try:
        # Convert string to matrix using ast.literal_eval for safe evaluation
        matrix = ast.literal_eval(matrix_string)
        if isinstance(matrix, list) and all(isinstance(row, list) for row in matrix):
            return matrix
    except:
        # Return an error or handle it as appropriate
        # print("Error converting string to matrix. Please check the format.")
        return None


def calculate_accuracy(ground_truth_str, prediction_str):
    """Calculate the element-wise accuracy from string inputs, returning 0% for any errors."""
    ground_truth = string_to_matrix(ground_truth_str)
    prediction = string_to_matrix(prediction_str)

    if ground_truth is None or prediction is None:
        return 0  # Treat conversion errors as 0% accuracy
    """Calculate the element-wise accuracy of prediction against ground truth."""
    if len(ground_truth) != len(prediction) or any(
            len(gt_row) != len(pred_row) for gt_row, pred_row in zip(ground_truth, prediction)):
        return 0

    if not (is_numeric_matrix(ground_truth) and is_numeric_matrix(prediction)):
        return 0

    # Flatten the matrices
    ground_truth_flat = [item for sublist in ground_truth for item in sublist]
    prediction_flat = [item for sublist in prediction for item in sublist]

    # Calculate matches and accuracy
    matches = sum(1 for gt, pred in zip(ground_truth_flat, prediction_flat) if gt == pred)
    accuracy = matches / len(ground_truth_flat)
    return accuracy


def calculate_accuracy_valid(ground_truth_str, prediction_str):
    """Calculate the element-wise accuracy from string inputs, returning 0% for any errors."""
    ground_truth = string_to_matrix(ground_truth_str)
    prediction = string_to_matrix(prediction_str)

    if ground_truth is None or prediction is None:
        return 'Invalid'  # Treat conversion errors as 0% accuracy
    """Calculate the element-wise accuracy of prediction against ground truth."""
    if len(ground_truth) != len(prediction) or any(
            len(gt_row) != len(pred_row) for gt_row, pred_row in zip(ground_truth, prediction)):
        return 'Invalid'

    if not (is_numeric_matrix(ground_truth) and is_numeric_matrix(prediction)):
        return 'Invalid'

    # Flatten the matrices
    ground_truth_flat = [item for sublist in ground_truth for item in sublist]
    prediction_flat = [item for sublist in prediction for item in sublist]

    # Calculate matches and accuracy
    matches = sum(1 for gt, pred in zip(ground_truth_flat, prediction_flat) if gt == pred)
    accuracy = matches / len(ground_truth_flat)
    return accuracy

## Evaluate general input prompt's result

# ARC
with open('results/hf_model_arc.json', 'r') as file:
    gpt = json.load(file)

questions = {}
root = 'data/ARC/training'
sub_files = list_files_sorted_by_size('data/ARC/training')
for sub_file in tqdm(sub_files):
    with open(root + '/' + sub_file, 'r') as file:
        question = json.load(file)
        questions[sub_file] = question

all_num = 0
correct_num = 0
incorrect_num = 0
p_accuracy = []
p_accuracy_valid = []
for index in gpt.keys():
    all_num += 1
    question = questions[index]
    gpt_str = gpt[index]
    gpt_str = gpt_str.replace('\n', '')

    if 'Output grid:' in gpt_str:
        gpt_str = gpt_str.split('Output grid:')[1]

    pattern = r"\[ ?\[(.*?)\] ?\]"

    try:
        gpt_str_use = re.findall(pattern, gpt_str)[0]
    except IndexError:
        p_accuracy.append(0)
        print(index)
        incorrect_num += 1
        continue
    gpt_str_use = '[[' + gpt_str_use + ']]'

    test_str = str(question['test'][0]['output'])
    p_accuracy.append(calculate_accuracy(test_str, gpt_str_use))
    if calculate_accuracy_valid(test_str, gpt_str_use) != 'Invalid':
        p_accuracy_valid.append(calculate_accuracy_valid(test_str, gpt_str_use))
    if (test_str == gpt_str_use):

        correct_num += 1
    else:

        incorrect_num += 1
        new_test_str = ''
        for i, string in enumerate(test_str):
            if string == ',':
                if test_str[i - 1] == ']':
                    new_test_str += '\n'
                else:
                    new_test_str += string
            else:
                new_test_str += string

        new_gpt_str = ''
        for i, string in enumerate(gpt_str_use):
            if string == ',':
                if gpt_str_use[i - 1] == ']':
                    new_gpt_str += '\n'
                else:
                    new_gpt_str += string
            else:
                new_gpt_str += string

        new_input = ''
        for i, string in enumerate(str(question['test'][0]['input'])):
            if string == ',':
                if str(question['test'][0]['input'])[i - 1] == ']':
                    new_input += '\n'
                else:
                    new_input += string
            else:
                new_input += string
## Acc
print(correct_num/all_num)
##Not M%
print((all_num-len(p_accuracy_valid))/all_num)

#ARAOC
task_type = ["move", "change_color", "copy", "mirror", "fill_internal", "scale"]
for task in task_type:
    with open('results/gpt_' + task + '.json',
              'r') as file:
        gpt = json.load(file)

    with open(
            'data/ARAOC/' + task + '.json',
            'r') as f:
        question_train = json.load(f)

    all_num = 0
    correct_num = 0
    incorrect_num = 0
    p_accuracy = []
    p_accuracy_valid = []
    for index in gpt.keys():
        all_num += 1
        # question = questions[index]
        question = question_train[int(index)]
        gpt_str = gpt[index]
        gpt_str = gpt_str.replace('\n', '')
        test_str = str(question['test'][0]['output'])

        pattern = r"\[ ?\[(.*?)\] ?\]"

        try:
            gpt_str_use = re.findall(pattern, gpt_str)[0]
        except IndexError:
            p_accuracy.append(0)
            print(index)
            incorrect_num += 1
            continue
        gpt_str_use = '[[' + gpt_str_use + ']]'

        pattern = r"\[ ?\[(.*?)\] ?\]"

        p_accuracy.append(calculate_accuracy(test_str, gpt_str_use))
        if calculate_accuracy_valid(test_str, gpt_str_use) != 'Invalid':
            p_accuracy_valid.append(calculate_accuracy_valid(test_str, gpt_str_use))
        if (test_str == gpt_str_use):
            correct_num += 1

        else:
            incorrect_num += 1
            new_test_str = ''
            for i, string in enumerate(test_str):
                if string == ',':
                    if test_str[i - 1] == ']':
                        new_test_str += '\n'
                    else:
                        new_test_str += string
                else:
                    new_test_str += string

            new_gpt_str = ''
            for i, string in enumerate(gpt_str_use):
                if string == ',':
                    if gpt_str_use[i - 1] == ']':
                        new_gpt_str += '\n'
                    else:
                        new_gpt_str += string
                else:
                    new_gpt_str += string

            new_input = ''
            for i, string in enumerate(str(question['test'][0]['input'])):
                if string == ',':
                    if str(question['test'][0]['input'])[i - 1] == ']':
                        new_input += '\n'
                    else:
                        new_input += string
                else:
                    new_input += string

    print(task)
    print('\n')
    ##ACC
    print(correct_num / all_num)
    print('\n')
    ##Not M%
    print((all_num-len(p_accuracy_valid))/all_num)

## Evaluate natural language input prompt's result
#ARAOC
task_type = ["move", "change_color", "copy", "mirror", "fill_internal", "scale"]
for task in task_type:
    with open('data/ARAOC' + task + '.json',
              'r') as file:
        question_train = json.load(file)

    with open(
            'results/gpt_' + task + '_language.json',
            'r') as f:
        gpt = json.load(f)

    all_num = 0
    correct_num = 0
    incorrect_num = 0
    p_accuracy = []
    p_accuracy_valid = []
    for index in gpt.keys():
        all_num += 1
        questions = question_train[int(index)]

        gpt_str = gpt[index]
        if gpt_str[-1] != ']':
            gpt_str += ']'
        if gpt_str[-2] == '.':
            gpt_str = gpt_str[:-2] + ']'

        try:
            gpt_str_use = str(convert_language_to_matrix(gpt_str))
        except IndexError:
            incorrect_num += 1
            p_accuracy.append(0)
            continue
        except ValueError:
            incorrect_num += 1
            p_accuracy.append(0)
            continue
        except SyntaxError:
            incorrect_num += 1
            p_accuracy.append(0)
            continue
        except NameError:
            incorrect_num += 1
            p_accuracy.append(0)
            continue

        test_str = str(questions['test'][0]['output'])
        p_accuracy.append(calculate_accuracy(test_str, gpt_str_use))
        if calculate_accuracy_valid(test_str, gpt_str_use) != 'Invalid':
            p_accuracy_valid.append(calculate_accuracy_valid(test_str, gpt_str_use))
        if (test_str == gpt_str_use):
            correct_num += 1
        else:
            incorrect_num += 1
    print(task)
    print('\n')
    ## ACC
    print(correct_num / all_num)
    print('\n')
    ## Not M%
    print((all_num - len(p_accuracy_valid)) / all_num)