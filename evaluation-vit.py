import os
import re
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import matplotlib.pyplot as plt
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Initialize the structure grid
def initialize_structure(W=20, H=16):
    return [[' ' for _ in range(W)] for _ in range(H)]

# Simulate placing a block
def drop_block(block_type, x_position, structure, W=20, H=16):
    if block_type == 'b11':
        block_height, block_width = 1, 1
    elif block_type == 'b31':
        block_height, block_width = 1, 3
    elif block_type == 'b13':
        block_height, block_width = 3, 1
    else:
        raise ValueError("Invalid block type")

    y_position = H - 1
    while y_position >= 0:
        collision = False
        for i in range(block_height):
            for j in range(block_width):
                slot = x_position + j - block_width // 2
                if not (0 <= slot < W) or (y_position - i < 0 or structure[y_position - i][slot] != ' '):
                    collision = True
                    break
            if collision:
                break
        if collision:
            break
        y_position -= 1

    final_y_position = y_position + 1
    for i in range(block_height):
        for j in range(block_width):
            slot = x_position + j - block_width // 2
            if 0 <= slot < W and 0 <= final_y_position - i < H:
                structure[final_y_position - i][slot] = '█'

# Visualize the structure grid
def visualize_structure(structure, save_path=None, W=20, H=16):
    visual_structure = np.array([[1 if cell != ' ' else 0 for cell in row] for row in structure])

    # Trim empty rows and columns
    non_zero_rows = np.any(visual_structure != 0, axis=1)
    non_zero_cols = np.any(visual_structure != 0, axis=0)
    trimmed_structure = visual_structure[non_zero_rows][:, non_zero_cols]

    plt.figure(figsize=(10, 8))
    plt.imshow(trimmed_structure, cmap="Greys", origin="lower")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# Load ViT model and evaluate scores
def evaluate_with_vit(image_path, target_letter):
    MODEL_NAME = 'pittawat/vit-base-uppercase-english-characters'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME,use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    softmax_outputs = torch.nn.Softmax(dim=-1)(logits)

    # Ensure the letter is uppercase
    target_letter = target_letter.upper()
    target_index = ord(target_letter) - ord('A')  # A=0, B=1, ..., Z=25

    # Extract confidence score for the target letter
    target_prob = softmax_outputs[0][target_index].item()
    return target_prob

# Preprocess the file content
def preprocess_file(commands):
    commands = commands[1:-1]  # Remove the first and last lines

    # Regex pattern to match valid drop_block lines
    valid_command_pattern = re.compile(r"^drop_block\('(b11|b31|b13)', \d+\)$")

    # Clean and validate each line
    cleaned_commands = []
    for command in commands:
        command = command.strip()  # Strip leading and trailing whitespace
        if not command or command.startswith("#"):  # Skip empty and comment lines
            continue

        # If the line is a drop_block command, remove comments
        if "drop_block" in command:
            command_no_comment = command.split("#")[0].strip()  # Remove anything after '#'
            if valid_command_pattern.match(command_no_comment):  # Validate format
                cleaned_commands.append(command_no_comment)

    return cleaned_commands

# Process generated result files
def process_results(input_folder, output_folder, target_letter):
    target_output_folder = os.path.join(output_folder, target_letter)
    os.makedirs(target_output_folder, exist_ok=True)
    result_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
    best_score = -1
    best_result_file = None
    best_cleaned_commands = None
    best_image_path = None

    for result_file in result_files:
        with open(os.path.join(input_folder, result_file), "r") as f:
            commands = f.readlines()

        # Preprocess the file content
        cleaned_commands = preprocess_file(commands)

        # Skip the file if no valid commands remain
        if not cleaned_commands:
            print(f"Skipped {result_file}: No valid commands after preprocessing.")
            continue

        # Initialize the structure grid
        structure = initialize_structure()
        for command in cleaned_commands:
            if "drop_block" in command:
                # Extract parameters and invoke the function
                match = re.match(r"^drop_block\('(b11|b31|b13)', (\d+)\)$", command)
                if match:
                    block_type = match.group(1)
                    x_position = int(match.group(2))
                    drop_block(block_type, x_position, structure, W=20, H=16)

        # Visualize and save the structure
        image_path = os.path.join(input_folder, f"{result_file.replace('.txt', '.png')}")
        visualize_structure(structure, save_path=image_path)

        # Evaluate with ViT model
        score = evaluate_with_vit(image_path, target_letter)
        print(f"Processed {result_file}: ViT Score for '{target_letter}' = {score:.4f}")

        # Record the best result
        if score > best_score:
            best_score = score
            best_result_file = result_file
            best_cleaned_commands = cleaned_commands
            best_image_path = image_path

    # Save the best result
    if best_result_file:
        with open(os.path.join(target_output_folder, best_result_file), "w") as f:
            f.write("'''\n")  # Opening '''
            f.writelines(f"{command}\n" for command in best_cleaned_commands)
            f.write("'''\n")  # Closing '''

        # Save the best image
        best_image_output_path = os.path.join(target_output_folder, os.path.basename(best_image_path))
        Image.open(best_image_path).save(best_image_output_path)

    print(f"Best result saved to {target_output_folder}, File: {best_result_file}, Score: {best_score:.4f}")

# Main entry point
if __name__ == "__main__":
    base_input_folder = "./tree_of_thought/raw"  # Replace with the root directory path
    output_folder = "./best_results"  # Replace with the directory to save the best results

    for folder_name in os.listdir(base_input_folder):
        folder_path = os.path.join(base_input_folder, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            process_results(folder_path, output_folder, target_letter=folder_name)

