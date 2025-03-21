# Durian Classifier

## Overview
This project implements a Convolutional Neural Network (CNN) model using TensorFlow and Keras to classify different types of durians (D13, D24, D197) from images. The model is trained on a dataset of durian images and evaluated using validation and test datasets.

## Project Structure
```
- durian_classifier/
  - durian_dataset/
    - train/
    - valid/
    - test/
  - scripts/
    - train_model.py
    - preprocess.py
  - models/
  - README.md
```

## Dependencies
Ensure you have the following Python libraries installed:

```sh
pip install numpy tensorflow matplotlib scikit-learn pillow
```

## Dataset
The dataset is organized into three directories:
- `train/`: Contains training images categorized into `D13`, `D24`, and `D197`.
- `valid/`: Contains validation images for model tuning.
- `test/`: Contains test images for final evaluation.

## Preprocessing
The images are preprocessed using TensorFlow's `ImageDataGenerator`, including:
- Resizing to 128x128 pixels
- Keeping aspect ratio
- Applying VGG16 preprocessing (RGB to BGR, zero-centering)

## Model Architecture
The CNN model consists of:
- 3 convolutional layers with ReLU activation and max-pooling
- Dropout layers to reduce overfitting
- Fully connected layers with 128, 64, and 32 neurons
- Output layer with 3 neurons (softmax activation for classification)

## Training
The model is compiled using:
- Adam optimizer with a learning rate of 0.0001
- Categorical cross-entropy loss function
- Accuracy as the evaluation metric

To train the model:
```python
model_info = cnn_model.fit(x=train_batches, validation_data=valid_batches, epochs=7, verbose=2)
```

## Evaluation
The model performance is visualized using accuracy and loss plots.

```python
plt.plot(model_info.history['accuracy'])
plt.plot(model_info.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'])
plt.show()
```

## Prediction
To make predictions on test data:
```python
test_imgs, test_labels = next(test_batches)
predictions = cnn_model.predict(test_batches, verbose=1)
```

## Future Improvements
- Implement data augmentation to improve generalization.
- Use a pre-trained model (e.g., ResNet, EfficientNet) for better feature extraction.
- Optimize hyperparameters for better performance.

## Author
Nathan Lim

