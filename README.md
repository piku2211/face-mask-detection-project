Face Mask Detection project:-

##Project Overview

This deep learning project detects whether a person is wearing a face mask or not using a **Convolutional Neural Network (CNN)**. The model is trained using the **Face Mask Detection** dataset from Kaggle and implemented in **Google Colab**. The final model can predict mask usage from a given image.



##Dataset

- Source: [Face Mask Detection Dataset - Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- Classes:
  - With Mask (Label = 1)
  - Without Mask (Label = 0)


##Dependencies

Make sure the following libraries are installed:

pip install numpy pandas matplotlib opencv-python tensorflow keras scikit-learn kaggle


Other tools:
- Google Colab (for development)
- Kaggle API (`kaggle.json` for dataset download)



##Step-by-Step Workflow

### Dataset Setup
- Dataset downloaded from Kaggle using `kaggle.json`.
- Extracted and stored in Google Drive.
- Two folders: with_mask and without_mask

###Preprocessing
- All images resized to 128x128
- Converted to NumPy arrays
- Mask images labeled as 1, without-mask images as 0
- Combined and split into training and testing sets
- Pixel values normalized by dividing by 255

###Model Building (CNN)
- Built using Keras Sequential API
- Layers:
  - Conv2D + ReLU
  - MaxPooling2D
  - Flatten
  - Dense + Dropout
  - Output layer with Softmax activation
- Compiled with:
  python
  optimizer='adam'
  loss='sparse_categorical_crossentropy'
  metrics=['accuracy']
  

###Model Training
python
model.fit(x_train_scaled, y_train, validation_split=0.1, epochs=10, batch_size=32)
- Training/Validation accuracy and loss plotted after training.

###Model Evaluation
- Evaluated on test data using model.evaluate()
- Saved final model using:
python
model.save("mask_detector_model.pkl")


###Prediction (Single Image)
- User inputs image path.
- Image resized, normalized, reshaped.
- Prediction is made using model.predict()
- Displays whether a mask is worn or not

##Output Example

- Input: Image of a person  
- Output:  
  - Prediction probabilities  
  - Final message:  
    -  Wearing a mask  
    -  Not wearing a mask


##Result

By following all the components, including preprocessing, training, and evaluation, the model successfully predicts mask usage with high accuracy.
