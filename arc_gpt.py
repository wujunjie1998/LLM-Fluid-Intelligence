from openai import OpenAI
import openai
import torch
import pdb
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

args = parser.parse_args()


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

if not os.path.exists('results'):
    os.makedirs('results')

original_results = {}
root = 'data/ARC/training'
sub_files = list_files_sorted_by_size(root)

## Use 100 ARC tasks for evaluation
with open('data/ARC/filename_100_tasks.json', 'r') as file:
    file_names = json.load(file)

for sub_file in tqdm(sub_files):
    if sub_file not in file_names:
        continue
    with open(root + '/' + sub_file, 'r') as file:
        question = json.load(file)

        input_prompt = io_only_prompt(question)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_prompt}]

        response = chatgpt_client.chat.completions.create(
            model='gpt model version',
            messages=messages,
            temperature=0.8,
        )

        original_results[sub_file] = response.choices[0].message.content

    with open('results/gpt_arc.json', 'w') as file:
        json.dump(original_results, file)
