# MoneyPlantDiseaseClassification
This repository contains a deep learning-based approach to classify money plant diseases into three categories: Bacterial Wilt Disease, Manganese Toxicity, and Healthy. The model is implemented using TensorFlow and Keras on Google Colab.

**Setup Instructions:**
  Prerequisites
  To run this project, you will need:
  Python 3.x
  TensorFlow: The main deep learning framework used for this model.
  Keras: Used for building and training the CNN model.
  Google Colab (optional): The code is designed to be run in Google Colab for convenience, but can also be run locally if you have the required dependencies installed.

**Dataset:**
The dataset consists of images of money plants categorized into three classes:
  1. Bacterial Wilt Disease
  2. Manganese Toxicity
  3. Healthy Plants
You will need to upload the dataset to Colab or set the correct paths if running locally.

**Setup Steps:**
Clone the repository:
git clone https://github.com/your-username/plant-disease-classification.git
cd plant-disease-classification

Upload dataset: Upload your dataset to the Colab environment or set the dataset path if running locally.
Open the notebook: Open the .ipynb notebook file in Google Colab or a Jupyter notebook environment.
Run the code: Follow the steps in the notebook to execute the model training and evaluation process.

**Execution Instructions:**
The model uses a Convolutional Neural Network (CNN) with 3 convolutional layers. The architecture includes the following layers:
  1. Conv2D (3 layers)
  2. MaxPooling2D
  3. Dropout for regularization
  4. Dense layers for final classification

**Data Preprocessing:**
  The dataset is preprocessed using ImageDataGenerator from Keras to augment images and prepare them for training.
  The model is trained using Adam Optimizer and sparse categorical crossentropy as the loss function.
  The model is evaluated on a test set and produces accuracy, precision, recall, and F1-score metrics.
  Run the training cell to start the training process. The model will automatically save the weights and the trained model.
  After training, run the evaluation cell to check model performance on the test data.

**Course Instructor Acknowledgment:**
This project is part of the research/assignment under the guidance of Md. Mynoddin. I would like to express my gratitude for his continuous support and feedback throughout this project.
