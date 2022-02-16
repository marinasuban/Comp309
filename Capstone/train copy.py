import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from skimage import io
from skimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening, selem

from keras_preprocessing.image import ImageDataGenerator

# Import modules
from tensorflow.keras import backend as K
import random
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16


# Code Reference
# https://outline.com/aJSrE5
# https://www.kaggle.com/tanyadayanand/end-to-end-pipeline-for-training-cnns-resnet/notebook

data_path = 'traindata'
image_category = ['cherry', 'strawberry', 'tomato']


# Load and preprocess data
def load_data():
    # split data into test and validation, normalise, augmentation
    data_gen = ImageDataGenerator(validation_split=0.2,
                                  height_shift_range=0.2,
                                  width_shift_range=0.2,
                                  rescale=1. / 255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
    # set images to 300x300
    valid = data_gen.flow_from_directory(directory=data_path, target_size=(300, 300),
                                         classes=image_category, batch_size=160, subset='validation')
    train = data_gen.flow_from_directory(directory=data_path, target_size=(300, 300),
                                         classes=image_category, batch_size=160, subset='training')

    return valid, train


# Get random image
def get_example(graph):
    paths = []
    for col in image_category:
        full_path = os.path.join(data_path, col, '*')

        # glob through the directory (returns a list of all file paths)
        class_paths = glob.glob(full_path)
        paths.append(class_paths)

        print(len(class_paths))
        # Random Drawing
        example_transforms(class_paths, graph)


# Plot image on graph
def plot_image(images, cmap=None):
    f, axes = plt.subplots(1, len(images), sharey=True)
    f.set_figwidth(15)

    for ax, image in zip(axes, images):
        ax.imshow(image, cmap)


# EDA visualisation - morphological transformations
def example_transforms(class_paths, transform):
    options = ["original", "channels", "threshold", "morphological", "normalize"]
    rand_index = random.randint(0, len(class_paths))
    image = io.imread(class_paths[rand_index])

    if transform == options[0]:  # Original
        print(image.shape)
        plt.imshow(image)
        plt.show()
    if transform == options[1]:  # Channels
        # plotting the original image and the RGB channels
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
        f.set_figwidth(15)
        ax1.imshow(image)
        # RGB channels
        # CHANNELID : 0 for Red, 1 for Green, 2 for Blue.
        ax2.imshow(image[:, :, 0])  # Red
        ax3.imshow(image[:, :, 1])  # Green
        ax4.imshow(image[:, :, 2])  # Blue
        f.suptitle('Different Channels of Image')
        plt.imshow(image)
        plt.show()
    if transform == options[2]:  # Threshold
        bin_image = image[:, :, 0] > 125

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.imshow(image)
        ax2.imshow(bin_image, cmap='gray')
        f.suptitle('Thresholded Image')
        plt.show()
    if transform == options[3]:  # Morphological  # Erosion, Dilation, Opening, Closing
        bin_image = image[:, :, 0] > 125
        # Disk of radius 3
        s = selem.disk(3)
        # Open & close
        open_img = binary_opening(bin_image, s)
        close_img = binary_closing(bin_image, s)

        # Erosion & Dilation
        eroded_img = binary_erosion(bin_image, s)
        dilated_img = binary_dilation(bin_image, s)

        plot_image([bin_image, open_img, close_img, eroded_img, dilated_img], cmap='gray')
        plt.imshow(image)
        plt.suptitle('Morphological Image')
        plt.show()
    if transform == options[4]:  # Normalize
        # Mode 1: Natural RGB images
        m1_image = image / 255
        # Mode 2: Medical/Non natural images
        m2_image = image - np.min(image) / np.max(image) - np.min(image)
        # Mode 3: Medical/Non natural images alt
        m3_image = image - np.percentile(image, 5) / np.percentile(image, 95) - np.percentile(image, 5)
        plot_image([image, m1_image, m2_image, m3_image], cmap='gray')
        plt.suptitle('Normalize Image')
        plt.show()


# Plot model result
def plot_result(history):
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.legend()
    plt.xlabel('No. epoch')
    plt.show()
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.legend()
    plt.xlabel('No. epoch')
    plt.show()


# Train MLP model
def MLP(valid, train):
    model = Sequential()
    model.add(Input(shape=(300, 300, 3)))
    model.add(Flatten())
    model.add(Dense(350, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train, epochs=50, verbose=1, validation_data=valid)
    plot_result(history)


# Train basic CNN model
def basic_CNN(valid, train):
    model = Sequential()
    model.add(Input(shape=(300, 300, 3)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    history = model.fit(train, epochs=25, verbose=1, validation_data=valid)
    plot_result(history)


# Train optimised CNN model
def Optimised_CNN(valid, train):
    model = Sequential()
    # 1st Convolution layer
    model.add(Input(shape=(300, 300, 3)))
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=2))
    # 2nd Convolution layer
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=2))
    # 3rd Convolution layer
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # 15% dropout
    model.add(Dropout(0.15))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=3))
    model.add(Flatten())
    # Classification layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.15))
    # Second dense layer with softmax activation function to return probabilities of classification
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    history = model.fit(train, epochs=50, verbose=1, validation_data=valid)
    plot_result(history)
    return model


# Train existing model
def Existing_model(valid, train):
    K.clear_session()
    # InceptionV3 model and use the weights from imagenet
    conv_base = VGG16(weights='imagenet', input_tensor=Input(shape=(300, 300, 3)),
                      include_top=False)

    VGG = conv_base.output
    pool = GlobalAveragePooling2D()(VGG)
    dense_1 = layers.Dense(512, activation='relu')(pool)
    output = layers.Dense(3, activation='softmax')(dense_1)

    # Define/Create the model for training
    model_VGG = models.Model(inputs=conv_base.input, outputs=output)
    # Compile the model with categorical crossentropy for the loss function and SGD for the optimizer with the learning
    model_VGG.compile(loss='categorical_crossentropy',
                      optimizer=SGD(learning_rate=1e-4, momentum=0.9),
                      metrics=['accuracy'])

    history = model_VGG.fit(train, epochs=50, verbose=1, validation_data=valid)
    plot_result(history)
    return model_VGG


if __name__ == '__main__':
    valid_data, train_data = load_data()
    # ------------------------------------------------------------------------#
    # Uncomment to view EDA - change param to below string for desired graph
    # "original", "channels", "threshold", "morphological", "normalize"
    # get_example("original")
    # ------------------------------------------------------------------------#
    # Uncomment to view Basic MLP
    # MLP(valid_data, train_data)
    # ------------------------------------------------------------------------#
    # Uncomment to view Basic CNN
    # basic_CNN(valid_data, train_data)
    # ------------------------------------------------------------------------#
    # Uncomment to view optimised CNN
    final_model = Optimised_CNN(valid_data, train_data)
    final_model.save("model.h5")
    print("Model Saved Successfully.")
    # ------------------------------------------------------------------------#
    # final_model = Existing_model(valid_data, train_data)
    # final_model.save("final.h5")
    # print("Model Saved Successfully.")
