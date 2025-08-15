https://huggingface.co/Thegame1161/Image_Classification
# Fine-Tuning a Vision Transformer (ViT) for Food Image Classification 🍔🍕

This repository contains an end-to-end project for fine-tuning a pre-trained Vision Transformer (ViT) model to classify images of food. The entire workflow is built using the Hugging Face ecosystem, including `datasets`, `transformers`, and `evaluate`, to create a powerful and accurate food classifier.

The project demonstrates the complete pipeline from loading data to training, evaluation, and finally, using the model for inference on new images.

## ✨ Key Features

  * **State-of-the-Art Model**: Utilizes `google/vit-base-patch16-224-in21k`, a powerful pre-trained Vision Transformer model, as the base for fine-tuning.
  * **Data Augmentation**: Employs `torchvision.transforms` to apply on-the-fly data augmentation (`RandomResizedCrop`), making the model more robust and preventing overfitting.
  * **End-to-End Workflow**: Covers every step of a computer vision project:
    1.  Data Loading
    2.  Image Preprocessing & Augmentation
    3.  Model Configuration
    4.  Training & Evaluation
    5.  Inference
  * **Hugging Face Integration**: Leverages the high-level `Trainer` API for a streamlined and efficient training process and the `pipeline` function for simple, production-ready inference.
  * **Custom Dataset**: Fine-tunes the model on the `AkshilShah21/food_images` dataset, which contains 10 different classes of food.

-----

## ⚙️ Project Workflow

The project follows a structured machine learning pipeline for computer vision:

1.  **Environment Setup**: Installs all necessary libraries, including `datasets`, `accelerate`, and `evaluate`.
2.  **Data Loading**: The `AkshilShah21/food_images` dataset is downloaded directly from the Hugging Face Hub.
3.  **Label Mapping**: Dictionaries (`label2id` and `id2label`) are created to map human-readable food labels (e.g., "pizza") to integer indices for the model.
4.  **Image Preprocessing**: An `AutoImageProcessor` is loaded from the ViT checkpoint to format images according to the model's specific requirements (size, normalization).
5.  **Data Augmentation**: A transformation pipeline using `torchvision` is defined to apply random crops, convert images to PyTorch tensors, and normalize them. This entire pipeline is applied to the dataset on-the-fly.
6.  **Model Configuration & Training**:
      * The pre-trained ViT model is loaded using `AutoModelForImageClassification`, with a new classification head configured for our specific number of food classes.
      * `TrainingArguments` are set up to define hyperparameters like learning rate, batch size, and number of epochs.
      * The `Trainer` API is used to handle the complete fine-tuning and evaluation loop.
7.  **Saving the Model**: The best-performing model from the training process is saved to a directory for future use.
8.  **Inference**: The fine-tuned model is loaded into an `image-classification` `pipeline` to easily classify new food images from a URL.

-----

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ and a CUDA-enabled GPU for faster training.

### Installation

1.  Clone the repository to your local machine:

    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  Install the required Python libraries:

    ```bash
    pip install datasets accelerate evaluate
    pip install torch torchvision
    pip install transformers
    ```

-----

## ▶️ How to Use the Trained Model

The fine-tuned model is saved in the `food_classification` directory. The easiest way to use it for prediction is with the Hugging Face `pipeline` function, which handles all the necessary preprocessing steps automatically.

```python
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO
import torch

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the saved model into an image-classification pipeline
classifier_pipe = pipeline("image-classification", model='food_classification', device=device)

# 2. Provide an image URL
image_url = 'https://www.indianhealthyrecipes.com/wp-content/uploads/2015/10/pizza-recipe-1.jpg'

# 3. Load the image and make a prediction
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

predictions = classifier_pipe(image)

# 4. Print the results
print(predictions)
```

### Expected Output

```
[{'score': 0.99..., 'label': 'pizza'}, 
 {'score': 0.00..., 'label': 'samosa'}, 
 {'score': 0.00..., 'label': 'omelette'}, 
 {'score': 0.00..., 'label': 'fried_rice'}, 
 {'score': 0.00..., 'label': 'donuts'}]
```

-----

## 📚 Dataset

This project uses the **`AkshilShah21/food_images`** dataset from the Hugging Face Hub. It contains images of 10 different types of food, already conveniently split into training and testing sets.

-----

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
