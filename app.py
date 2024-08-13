# %% [code] {"execution":{"iopub.status.busy":"2024-08-05T04:39:39.598367Z","iopub.execute_input":"2024-08-05T04:39:39.599077Z","iopub.status.idle":"2024-08-05T04:39:59.769586Z","shell.execute_reply.started":"2024-08-05T04:39:39.599026Z","shell.execute_reply":"2024-08-05T04:39:59.768270Z"}}
# import all neccessary libraries
import numpy as np
import tensorflow as tf
print("Tensorflow version: {}".format(tf.__version__))
from tensorflow import keras
print("Keras version: {}".format(keras.__version__))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from PIL import ImageOps, Image
import itertools
# import os
# import shutil
# import random
# import glob
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-05T04:39:59.771799Z","iopub.execute_input":"2024-08-05T04:39:59.772667Z","iopub.status.idle":"2024-08-05T04:39:59.778879Z","shell.execute_reply.started":"2024-08-05T04:39:59.772618Z","shell.execute_reply":"2024-08-05T04:39:59.777246Z"}}
# defining file paths
train_path = '/Users/nathan/Documents/github_repo/durian_classifier/durian_dataset/train'
valid_path = '/Users/nathan/Documents/github_repo/durian_classifier/durian_dataset/valid'
test_path = '/Users/nathan/Documents/github_repo/durian_classifier/durian_dataset/test'

# %% [code] {"execution":{"iopub.status.busy":"2024-08-05T04:39:59.783249Z","iopub.execute_input":"2024-08-05T04:39:59.784819Z","iopub.status.idle":"2024-08-05T04:39:59.958531Z","shell.execute_reply.started":"2024-08-05T04:39:59.784770Z","shell.execute_reply":"2024-08-05T04:39:59.957184Z"}}
# preprocessing photos (fit into target_size and keep_aspect_ratio of the photo. Crop when neccessary.preprocessing photos
# Then apply vgg16 which changed RGB to BGR and then zero-center each colour channel wrt to ImageNet dataset without scaling)
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(128,128), classes=['D13','D24','D197'], keep_aspect_ratio=True, batch_size=32)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(128,128), classes=['D13','D24','D197'], keep_aspect_ratio=True, batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(128,128), classes=['D13','D24','D197'], keep_aspect_ratio=True, batch_size=5, shuffle=False)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-05T04:39:59.960368Z","iopub.execute_input":"2024-08-05T04:39:59.960835Z","iopub.status.idle":"2024-08-05T04:39:59.967781Z","shell.execute_reply.started":"2024-08-05T04:39:59.960796Z","shell.execute_reply":"2024-08-05T04:39:59.966531Z"}}
# # data processing to greyscale the image instead of vgg16
# #define function to greyscale the photos
# def to_grayscale_then_rgb(image):
#     image = tf.image.rgb_to_grayscale(image)
#     image = tf.image.grayscale_to_rgb(image)
#     return image

# # preprocessing photos (fit into target_size and keep_aspect_ratio of the photo. Crop when neccessary.preprocessing photos
# # Then apply greyscale the photos using the to_grayscale_then_rgb function
# train_batches = ImageDataGenerator(rescale=1/255,
#                                    preprocessing_function=to_grayscale_then_rgb) \
#     .flow_from_directory(directory=train_path, target_size=(128,128), classes=['D13','D24','D197'], batch_size=32, keep_aspect_ratio=True)
# valid_batches = ImageDataGenerator(rescale=1/255,
#                                    preprocessing_function=to_grayscale_then_rgb) \
#     .flow_from_directory(directory=valid_path, target_size=(128,128), classes=['D13','D24','D197'], batch_size=10, keep_aspect_ratio=True)
# test_batches = ImageDataGenerator(rescale=1/255,
#                                    preprocessing_function=to_grayscale_then_rgb) \
#     .flow_from_directory(directory=test_path, target_size=(128,128), classes=['D13','D24','D197'], batch_size=5, keep_aspect_ratio=True, shuffle=False)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-05T04:39:59.969947Z","iopub.execute_input":"2024-08-05T04:39:59.970503Z","iopub.status.idle":"2024-08-05T04:39:59.986834Z","shell.execute_reply.started":"2024-08-05T04:39:59.970444Z","shell.execute_reply":"2024-08-05T04:39:59.985393Z"}}
# reafirm the correct total size for each batch
assert train_batches.n == 672
assert valid_batches.n == 213
assert test_batches.n == 108
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 3

# %% [code] {"execution":{"iopub.status.busy":"2024-08-05T04:39:59.988565Z","iopub.execute_input":"2024-08-05T04:39:59.989136Z","iopub.status.idle":"2024-08-05T04:40:00.338676Z","shell.execute_reply.started":"2024-08-05T04:39:59.989087Z","shell.execute_reply":"2024-08-05T04:40:00.336832Z"}}
# labels were defined in the train_batches, valid_batches and test_batches objects using the ImageDataGenerator function of keras.preprocessing.image
# Extract the durian photos (in the form of numpy array) and their respective One Hot Encoded labels into imgs and labels variables respectively from train_batches.
imgs, labels = next(train_batches)

# print the array shape of the first durian photo
print(imgs[0].shape)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-05T04:40:00.341020Z","iopub.execute_input":"2024-08-05T04:40:00.341601Z","iopub.status.idle":"2024-08-05T04:40:00.351210Z","shell.execute_reply.started":"2024-08-05T04:40:00.341533Z","shell.execute_reply":"2024-08-05T04:40:00.349496Z"}}
# define plotImages function which will be used to print out the processed durian images of the imgs numpy array
def plotImages(images_arr):
    fig, axes = plt.subplots(1,10,figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-08-05T04:40:00.352962Z","iopub.execute_input":"2024-08-05T04:40:00.353404Z","iopub.status.idle":"2024-08-05T04:40:01.511792Z","shell.execute_reply.started":"2024-08-05T04:40:00.353367Z","shell.execute_reply":"2024-08-05T04:40:01.509694Z"}}
# print the processed durian images and their respective One Hot Encoded labels
plotImages(imgs)
print(labels)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-05T04:40:01.513841Z","iopub.execute_input":"2024-08-05T04:40:01.515321Z","iopub.status.idle":"2024-08-05T04:40:02.115703Z","shell.execute_reply.started":"2024-08-05T04:40:01.515264Z","shell.execute_reply":"2024-08-05T04:40:02.113832Z"}}
#defining deep learning structure
# defining a Sequential object
cnn_model=Sequential()

# creating CNN
#adding 1st convolution layer
cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3),activation='relu',input_shape=imgs[0].shape))
cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
#adding pooling layer
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.1))

#2nd convolution layer which is similar to 1st layer except wo the input shape and filter is 64
cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
#adding pooling layer
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.1))

#3rd convolution layer which is similar to 1st layer except wo the input shape and filter is 32
cnn_model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
#adding pooling layer
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.1))

#adding fully connected layer
cnn_model.add(Flatten())

# creating ANN
#Add the fully connected ANN with 128 neurons
cnn_model.add(Dense(256,activation='relu'))
cnn_model.add(Dense(128,activation='relu'))
cnn_model.add(Dense(64,activation='relu'))
cnn_model.add(Dense(32,activation='relu'))
#adding output layer
cnn_model.add(Dense(3,activation='softmax'))

# printing the deep learning structrure
cnn_model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2024-08-05T04:40:02.121611Z","iopub.execute_input":"2024-08-05T04:40:02.122189Z","iopub.status.idle":"2024-08-05T04:40:02.151112Z","shell.execute_reply.started":"2024-08-05T04:40:02.122143Z","shell.execute_reply":"2024-08-05T04:40:02.148853Z"}}
# compiling the deep learning model with categorical_crossentroy loss function, Adam (a type of stochastic gradient descent) for backpropagation,
# and using accuracy as the metrics
cnn_model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

# %% [code] {"execution":{"iopub.status.busy":"2024-08-05T04:40:02.153386Z","iopub.execute_input":"2024-08-05T04:40:02.153931Z"}}
# building the deep learning model (training the model)
model_info = cnn_model.fit(x=train_batches, validation_data=valid_batches, epochs =7, verbose=2)

# %% [code]
# list all data in history
print(model_info.history)

# summarize history for accuracy and plotting out the accuracy graph
plt.plot(np.arange(7)+1,model_info.history['accuracy'])
plt.plot(np.arange(7)+1,model_info.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()
# summarize history for loss and plotting out the loss graph
plt.plot(np.arange(7)+1,model_info.history['loss'])
plt.plot(np.arange(7)+1,model_info.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

# %% [code]
# labels were defined in the train_batches, valid_batches and test_batches objects using the ImageDataGenerator function of keras.preprocessing.image
# Extract the next durian photos (in the form of numpy array) and their respective One Hot Encoded labels into imgs and labels variables respectively from test_batches.
test_imgs, test_labels = next(test_batches)
plotImages(test_imgs)
print(test_labels)

# %% [code]
# printing Ordinal labels for the test_batches
test_batches.classes

# %% [code]
# making prediction using the deep learning model from the test_batches
# predictions will be an array of probabilities
predictions = cnn_model.predict(x=test_batches, verbose=0)

# %% [code]
# rounding off the probabilities in the prediction array of probabilities
np.round(predictions)

# %% [code]
# building confusion matrix and pass it into cm variable
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

# %% [code]
# defining plot_confusion_matrix to plot the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalized=True'.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float')/ cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i,j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# %% [code]
# printing the indices of each category
test_batches.class_indices

# %% [code]
# ploting the confusion matrix
cm_plot_labels = ["D13","D24","D197"]
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

# %% [code]
# saving the deep learning model
cnn_model.save('durian_classification_trained_model.h5')

# %% [code]
# testing model from unseen data
#Step 1
# my_image = plt.imread(os.path.join('uploads', filename))
# img = plt.imread("C:\\Users\\tanke\\Python_Projects\\Project_Learn_Deep_Learning\\durians_dataset\\D13\\P_20200815_150324.jpg")
img = Image.open("/Users/nathan/Documents/github_repo/durian_classifier/durian_dataset/for testing/D197/20240808_1.jpg")
print(img)
#Step 2
my_image = ImageOps.fit(img, (128,128))
print(np.array(my_image)[0,:].shape)
my_image_re = tf.keras.applications.vgg16.preprocess_input(np.array(my_image))
f, axarr = plt.subplots(1,3)
axarr[0].imshow(img)
axarr[1].imshow(my_image)
axarr[2].imshow(my_image_re)

#Step 3
#with graph.as_default():
    #set_session(sess)
    #Add
cnn_model.run_eagerly=True  
probabilities = cnn_model.predict(np.array([my_image_re,]), verbose=0)[0,:]
print(probabilities)
#Step 4
number_to_class = ['D13','D24','D197']
index = np.argsort(probabilities)
predictions = {
    "class1":number_to_class[index[2]],
    "class2":number_to_class[index[1]],
    "class3":number_to_class[index[0]],
    "prob1":probabilities[index[2]],
    "prob2":probabilities[index[1]],
    "prob3":probabilities[index[0]],
    }
print(predictions)

# %% [code]
#for predicting using model that was trained by greyscale images data

# #Step 1
# # my_image = plt.imread(os.path.join('uploads', filename))
# # img = plt.imread("C:\\Users\\tanke\\Python_Projects\\Project_Learn_Deep_Learning\\durians_dataset\\D13\\P_20200815_150324.jpg")
# img = Image.open("C:\\Users\\tanke\\Python_Projects\\Project_Learn_Deep_Learning\\durians_dataset\\D24\\P_20200817_114123_SRES.jpg")
# print(img)
# #Step 2
# my_image = ImageOps.fit(img, (128,128))
# print(np.array(my_image)[0,:].shape)
# my_image_rescale = np.array(my_image)*1./255
# my_image_greyscale = to_grayscale_then_rgb(my_image_rescale)
# f, axarr = plt.subplots(1,4)
# axarr[0].imshow(img)
# axarr[1].imshow(my_image)
# axarr[2].imshow(my_image_rescale)
# axarr[3].imshow(my_image_greyscale)

#Step 3
#with graph.as_default():
    #set_session(sess)
    #Add
# cnn_model.run_eagerly=True  
# probabilities = cnn_model.predict(np.array([my_image_re,]), verbose=0)[0,:]
# print(probabilities)
# #Step 4
# number_to_class = ['D13','D24','D197']
# index = np.argsort(probabilities)
# predictions = {
#     "class1":number_to_class[index[2]],
#     "class2":number_to_class[index[1]],
#     "class3":number_to_class[index[0]],
#     "prob1":probabilities[index[2]],
#     "prob2":probabilities[index[1]],
#     "prob3":probabilities[index[0]],
#     }
# print(predictions)