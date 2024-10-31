# College Exercise: TensorFlow Projects

This repository contains two projects demonstrating fundamental TensorFlow-based machine learning and mathematical operations. Each project is organized in its own directory, with individual files explaining important functionality.

## Table of Contents
- [Project 1: MNIST Digit Classifier](#project-1-mnist-digit-classifier)
- [Project 2: TensorFlow Math Functions](#project-2-tensorflow-math-functions)

---

## Project 1: MNIST Digit Classifier

**Folder:** `Lab1`

### Overview
This project builds a simple neural network using TensorFlow to recognize handwritten digits from the MNIST dataset. The model is saved to a file after training to avoid retraining on subsequent runs. The code includes a function to load an external image and classify it as one of the digits (0-9).

### Features
1. **Model Saving & Loading**: The model is saved to `model.keras` after training. If the file exists, it is loaded instead of retraining.
2. **Image Preprocessing**: The code uses PIL to preprocess and invert an input image for model compatibility.
3. **Digit Prediction**: After loading a preprocessed image, the model outputs a prediction showing the recognized digit.

### Usage
1. Run the code with Python.
2. To test the model with an image, save a digit image as `three.png` (or another name) and place it in the root directory.
3. Modify `img_path` in the script to point to the image filename.

### Key Files
- `model.keras`: Contains the trained model (created after the initial run).
- `three.png`: Sample image file for testing prediction.

---

## Project 2: TensorFlow Math Functions

**Folder:** `Lab2`

### Overview
This project contains TensorFlow functions for various mathematical computations. These include point rotation on a 2D plane, angle-based point rotation, solving linear equation systems using matrix methods, and a command-line interface for matrix input.

### Tasks
1. **Rotate Point**: Rotates a 2D point around the origin by a specified angle.
2. **Solve Linear System**: Solves a linear equation system `Ax = B` using matrix operations.
3. **Command Line Interface**: The main program allows users to input a linear equation system as matrices through the command line.

### Usage
1. **Command Line Usage**:
    ```shell
    python TensorMatrix.py '[[matrix_coefficients]]' '[[results_vector]]'
    ```
   - For example, to solve `2x + 3y = 8` and `x - y = 2`, use:
     ```shell
     python TensorMatrix.py '[[2.0, 3.0], [1.0, -1.0]]' '[[8.0], [2.0]]'
     ```

### Key Files
- `TensorMatrix.py`: Core file with rotation functions, linear system solver, and CLI setup.
