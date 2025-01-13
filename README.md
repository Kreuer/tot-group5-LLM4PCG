# tot-group5-LLM4PCG
# README

## Overview
This project implements a pipeline for generating, visualizing, and evaluating structures based on block placement commands. It uses a Vision Transformer (ViT) model to assess the similarity between generated structures and predefined targets (letters A-Z).

## Features
- **Structure Initialization**: Creates a grid for block placement.
- **Block Placement Simulation**: Simulates dropping blocks of various shapes (`b11`, `b31`, `b13`) onto the grid.
- **Visualization**: Generates images of the grid after block placement.
- **Evaluation**: Uses the `pittawat/vit-base-uppercase-english-characters` ViT model to evaluate the similarity between generated structures and target letters.
- **Result Management**: Identifies and saves the best-matching structure for each target letter.

## Dependencies
- Python 3.11+
- PyTorch
- Transformers
- NumPy
- Matplotlib
- Pillow

Install the dependencies using:
```bash
pip install torch transformers numpy matplotlib pillow
```

## Usage

### Input Structure
The input folder structure should be organized as follows:
```
./tree_of_thought2/raw/
    A/
        result1.txt
        result2.txt
    B/
        result1.txt
        result2.txt
    ...
```
Each `.txt` file contains block placement commands for a specific letter.

### Running the Script
To process all results and save the best structures for each letter, execute:
```bash
python evaluation-vit.py
```

### Output Structure
The output folder will be organized as follows:
```
./best_results/
    A/
        best_result.txt
        best_result.png
    B/
        best_result.txt
        best_result.png
    ...
```
Each `.txt` file contains the cleaned and validated block placement commands, and each `.png` file visualizes the best structure.

## Key Functions

### `initialize_structure(W=20, H=16)`
Initializes a grid with dimensions `W` (width) and `H` (height).

### `drop_block(block_type, x_position, structure, W=20, H=16)`
Simulates the placement of a block on the grid.

### `visualize_structure(structure, save_path=None, W=20, H=16)`
Generates a visual representation of the grid and saves it as an image.

### `evaluate_with_vit(image_path, target_letter)`
Uses the ViT model to compute a similarity score between the generated structure and the target letter.

### `process_results(input_folder, output_folder, target_letter)`
Processes all `.txt` files in the `input_folder` for a specific `target_letter`. Identifies and saves the best structure.

## Notes
- The `evaluate_with_vit` function requires a pre-trained ViT model (`pittawat/vit-base-uppercase-english-characters`). Ensure it is available locally or via the internet.
- The script suppresses `FutureWarning` messages for a cleaner output.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

