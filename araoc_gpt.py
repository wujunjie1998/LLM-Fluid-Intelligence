from openai import OpenAI
import openai
import torch
import numpy as np
import json
import random
import os
from tqdm import tqdm
import argparse

## Import OpenAI Key
openai.organization = "your openai organization"
chatgpt_client = OpenAI(api_key='your openai key')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(2024)

parser = argparse.ArgumentParser(description="Example script to demonstrate command line argument parsing.")


# Add arguments
parser.add_argument("--task_type", type=str, required=True, help="The task type (e.g., 'move')")
parser.add_argument("--language", action='store_true', required=False, help="Use natural language input")
parser.add_argument("--matrix", action='store_true', required=False, help="Ask questions about input matrix properties")

args = parser.parse_args()

task_type = args.task_type


preamble = """
Lets play a game where you are transforming an input grid of numbers into an output grid of numbers.

The numbers represent different colors:
0 = black
1 = blue
2 = red
3 = green
4 = yellow
5 = gray
6 = magenta
7 = orange
8 = cyan
9 = brown

"""

def io_only_prompt(task):
    pre_prompt = "You will be playing a game that need to find common patterns from input examples and apply the pattern for prediction on new examples."
    input_output_example = "\n\nHere are examples of input grids and its corresponding output grids:\n"

    for example_id in range(len(task['train'])):
        train_input = task['train'][example_id]['input']
        train_output = task['train'][example_id]['output']

        input_output_example += "Example input grid:\n" + str(train_input) + "\nExample output grid:\n" + str(
            train_output) + "\n\n"

    input_grid = task['test'][0]['input']

    prompt = pre_prompt + preamble + input_output_example + "\n\nThe input grid is:\n" + str(
        input_grid) + "\n\nWhat is the output grid? Please only output your answer without analysis in the following format:\nOutput grid:"
    return prompt

def convert_matrix_to_language(matrix):
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    # Find non-zero elements and their coordinates, adjusted for a bottom-left origin
    non_zero_elements = []
    for y in range(rows):
        for x in range(cols):
            if matrix[y][x] != 0:
                # Calculate Cartesian coordinates from matrix indices
                cartesian_y = (rows - 1) - y
                non_zero_elements.append(f"({matrix[y][x]}, ({x}, {cartesian_y}))")

    # Create the final description string
    final_description = (
        f"The matrix dimensions are {cols} columns by {rows} rows. "
        f"Coordinates are based on a Cartesian coordinate system with the origin (0,0) at the bottom-left corner. "
        f"The coordinates of the non-zero elements, listed from top to bottom and left to right, are: [{', '.join(non_zero_elements)}]"
    )

    return final_description

def io_only_prompt_natural_language(task):
    input_output_example = "\n\nHere are examples of input grids and its corresponding output grids:\n"

    for example_id in range(len(task['train'])):
        train_input = convert_matrix_to_language(task['train'][example_id]['input'])
        train_output = convert_matrix_to_language(task['train'][example_id]['output'])

        input_output_example += "Example input grid:\n" + str(train_input) + "\nExample output grid:\n" + str(
            train_output) + "\n\n"

    input_grid = convert_matrix_to_language(task['test'][0]['input'])

    prompt = preamble + input_output_example + "\n\nThe input grid is:\n" + str(
        input_grid) + "\n\nWhat is the output grid? Please answer in the following format without outputting analysis:\nOutput grid:"
    return prompt

def matrix_prompt(task):
    instruction = """
Given a matrix in the format of numpy array, please answer the following questions:\n1. What is the size of this matrix?  Output in the format of (a,b).\n2. What is the location of the non-zero subgrids. Please first find out all the corner elements of the subgrids, then output their locations in the order of [top-left, top-right, bottom-left, bottom-right], in the format of (which row, which col).\n3. What is the transpose of this matrix? Output the transposed matrix in the format of a numpy array with elements separated by commas and enclosed in square brackets for each row like "[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]".\n\nPlease only output your answer without analysis in the following format:\n1.Size: \n2.Location: \n3.Transpose:\n\nInput Matrix: 
"""
    input_grid = task['test'][0]['input']

    prompt = instruction + str(
        input_grid)
    return prompt

if not os.path.exists('results'):
    os.makedirs('results')

original_results = {}
## import test file
with open('data/ARAOC/'+task_type+'.json', 'r') as f:
    question_train = json.load(f)

for index, question in tqdm(enumerate(question_train[:100])):
    input_prompt = io_only_prompt(question)
    if args.language:
        input_prompt = io_only_prompt_natural_language(question)
    else:
        if args.matrix:
            input_prompt = matrix_prompt(question)
        else:
            input_prompt = io_only_prompt(question)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_prompt}]

    response = chatgpt_client.chat.completions.create(
        model='gpt model version',
        messages=messages,
        temperature=0.8,
    )

    original_results[str(index)] = response.choices[0].message.content

    if args.language:
        with open('results/gpt_' + task_type + '_language.json', 'w') as file:
            json.dump(original_results, file)
    elif args.matrix:
        with open('results/gpt_' + task_type + '_matrix.json', 'w') as file:
            json.dump(original_results, file)
    else:
        with open('results/gpt_'+task_type+'.json', 'w') as file:
            json.dump(original_results, file)
