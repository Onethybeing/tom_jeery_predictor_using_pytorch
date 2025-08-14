# Tom & Jerry Image Classifier
<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/aadc9f40-94a2-443e-8540-b1cf2fd0ce90" />

A PyTorch-based image classification project to detect characters from the "Tom and Jerry" cartoon. This model identifies whether an image contains Tom, Jerry, both, or neither.


https://github.com/user-attachments/assets/567a884b-c02f-4b11-9d0d-6b60655e7623


-----

## Table of Contents

  - [Project Overview](https://www.google.com/search?q=%23project-overview)
  - [Live Demo / Screenshot](https://www.google.com/search?q=%23live-demo--screenshot)
  - [Key Features](https://www.google.com/search?q=%23key-features)
  - [Technology Stack](https://www.google.com/search?q=%23technology-stack)
  - [Project Structure](https://www.google.com/search?q=%23project-structure)
  - [Installation Guide](https://www.google.com/search?q=%23installation-guide)
  - [How to Use](https://www.google.com/search?q=%23how-to-use)
      - [Training the Model](https://www.google.com/search?q=%231-training-the-model)
      - [Making Predictions](https://www.google.com/search?q=%232-making-predictions)
  - [Model Performance](https://www.google.com/search?q=%23model-performance)
  - [Contributing](https://www.google.com/search?q=%23contributing)
  - [License](https://www.google.com/search?q=%23license)
  - [Acknowledgements](https://www.google.com/search?q=%23acknowledgements)

## Project Overview

This project provides an end-to-end solution for a classic computer vision problem: image classification. It leverages a state-of-the-art deep learning model, **EfficientNet-B0**, which is pre-trained on the ImageNet dataset and then fine-tuned on a custom "Tom and Jerry" dataset. The goal is to accurately categorize any given image into one of four classes: `tom`, `jerry`, `both` characters present, or `neither`.

The repository contains the complete workflow, from the Jupyter Notebook used for data preprocessing and model training to a user-friendly Python script for running inference on new, unseen images.

## Live Demo / Screenshot

Here is an example of the prediction script in action:

*Note: This is a placeholder image. You can replace it with a screenshot of your script's output.*

## Key Features

  - **High Accuracy**: Utilizes transfer learning with EfficientNet for robust performance.
  - **Multi-Class Classification**: Distinguishes between 4 distinct categories.
  - **Command-Line Interface**: Predict on any image directly from the terminal.
  - **Visual Feedback**: Displays the input image with the predicted label and confidence score.
  - **Reproducible Training**: The included Jupyter Notebook allows for easy retraining and experimentation.

## Technology Stack

  - **Core Framework**: Python 3.9+
  - **Deep Learning**: PyTorch, Torchvision
  - **Data Handling**: Pillow (PIL)
  - **Visualization**: Matplotlib
  - **Development Environment**: Jupyter Notebook

## Project Structure

```
tom-jerry-classifier/
│
├── tom_and_jerry_model.pth     # Trained PyTorch model file
├── Untitled0.ipynb             # Jupyter Notebook for training and experimentation
├── predict.py                  # Script for making predictions on new images
├── README.md                   # You are here
└── requirements.txt            # List of Python dependencies
```

## Installation Guide

Follow these steps to set up the project environment on your local machine.

1.  **Prerequisites**:

      - Python 3.9 or higher
      - `pip` package manager

2.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/tom-jerry-classifier.git
    cd tom-jerry-classifier
    ```

3.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install Dependencies from `requirements.txt`:**
    A `requirements.txt` file is provided for easy installation.

    ```bash
    pip install -r requirements.txt
    ```

    *(If you don't have a `requirements.txt` file, you can create one with `pip freeze > requirements.txt` after installing the packages below)*

    ```bash
    pip install torch torchvision matplotlib Pillow
    ```

## How to Use

### 1\. Training the Model

The file `Untitled0.ipynb` contains all the steps for training the model from scratch.

  - **Dataset**: Before running, ensure your dataset is structured correctly as required by `ImageFolder`. Place your images in a `data` directory as follows:
    ```
    data/
    ├── train/
    │   ├── tom/
    │   ├── jerry/
    │   ├── both/
    │   └── neither/
    └── valid/
        ├── tom/
        ├── jerry/
        ├── both/
        └── neither/
    ```
  - **Execution**: Open and run the cells in the Jupyter Notebook. The notebook will handle data loading, model fine-tuning, and will save the final trained model as `tom_and_jerry_model.pth`.

### 2\. Making Predictions

The `predict.py` script is designed to be run from the command line. It loads the trained model and classifies an image you provide.

  - **Command**:
    ```bash
    python predict.py path/to/your/image.jpg
    ```
  - **Example**:
    ```bash
    python predict.py assets/test-image-1.png
    ```
  - **Output**: The script will first print the prediction details to your terminal and then display the image in a new window with the prediction as the title.
    ```
    ------------------------------
    Results for: test-image-1.png
      -> Predicted Class: both
      -> Confidence: 0.9872
    ------------------------------
    ```

*For this to work, your `predict.py` should be updated to use command-line arguments. Here is the recommended code for `predict.py`:*

\<details\>
\<summary\>Click to see the recommended \<b\>predict.py\</b\> code\</summary\>

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import argparse

# --- CONFIGURATION BASED ON YOUR NOTEBOOK ---
CLASS_NAMES = ['both', 'jerry', 'neither', 'tom']
predict_transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_image(model_path, image_path, transform, class_names, device):
    """Loads a trained model and predicts the class of a new image."""
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.to(device)
        model.eval()
    except Exception as e:
        return {"error": f"Error loading model: {e}"}

    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        return {"error": f"Error processing image: {e}"}

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, 1)
        predicted_class_name = class_names[predicted_index.item()]
        confidence_score = confidence.item()

    result = {
        "predicted_class": predicted_class_name,
        "confidence": f"{confidence_score:.4f}"
    }
    return result

def print_and_show(prediction, image_path):
    """Prints results and displays the image."""
    print("-" * 30)
    print(f"Results for: {os.path.basename(image_path)}")
    if "error" in prediction:
        print(prediction["error"])
        return
    else:
        print(f"  -> Predicted Class: {prediction['predicted_class']}")
        print(f"  -> Confidence: {prediction['confidence']}")
    print("-" * 30)

    try:
        image = Image.open(image_path)
        plt.figure(figsize=(8, 8))
        title = (
            f"Predicted: {prediction['predicted_class']}\n"
            f"Confidence: {prediction['confidence']}"
        )
        plt.imshow(image)
        plt.title(title, fontsize=12)
        plt.axis('off')
        plt.show()
    except FileNotFoundError:
        print(f"\nCould not display image. File not found at: {image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Tom & Jerry characters in an image.")
    parser.add_argument("image_path", help="The path to the image file.")
    parser.add_argument("--model_path", default="tom_and_jerry_model.pth", help="Path to the trained model file.")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at '{args.model_path}'")
    elif not os.path.exists(args.image_path):
        print(f"Error: Image not found at '{args.image_path}'")
    else:
        prediction_result = predict_image(
            model_path=args.model_path,
            image_path=args.image_path,
            transform=predict_transform,
            class_names=CLASS_NAMES,
            device=DEVICE
        )
        print_and_show(prediction_result, args.image_path)
```

\</details\>

## Model Performance

The model was trained for 10 epochs, achieving the following performance on the final epoch:

  - **Training Accuracy**: \~91.5%
  - **Validation Accuracy**: **\~88.8%**
  - **Validation Loss**: \~0.35

*(Note: These are example values based on a typical training run. Actual performance may vary.)*

## Contributing

Contributions are welcome\! If you have suggestions for improvements, please feel free to fork the repository and submit a pull request. You can also open an issue with the "enhancement" tag.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

This project is distributed under the MIT License. See `LICENSE.txt` for more information.

## Acknowledgements
for more information contact at onethybeing@gmail.com
