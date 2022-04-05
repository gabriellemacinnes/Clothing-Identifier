# import statements
import sys
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


from matplotlib import pyplot as plt

# to log, pass the string "log" as arg when running the file
DO_LOG = "log" in sys.argv

# training of dataset
labels = pd.read_csv('data/train_labels.csv')  # read labels
train_images = np.load('data/train_images.npy') # load train images

# mapping
label_mapping = pd.read_csv('data/label_int_to_str_mapping.csv')  # read label mapping

# testing part of dataset
sample_submission = pd.read_csv('data/sample_submission.csv') # read sample_submission
test_images = np.load('data/test_images.npy') # load test images

# Normalize the data
train_images_norm = train_images / 255
test_images_norm = test_images / 255


if (DO_LOG):
    print(f"""
    {labels.shape= }
    {train_images.shape= }
    {label_mapping.shape= }
    {sample_submission.shape= }
    {test_images.shape= }

    Before normalization:
    {train_images[0,0:5,0:5]}

    After normalization:
    {train_images_norm[0,0:5,0:5]}
    """)

    # plot the first 9 images
    f, axarr = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            axarr[i, j].imshow(train_images[i*3+j].reshape(28, 28), cmap='gray')
            axarr[i, j].set_title(f"Train Image {i*3+j}")
            axarr[i, j].axis('off')

    plt.show()

def define_model():
    """
    This function defines the CNN model.
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_images, train_labels, label_mapping):
    """
    This function trains the model.
    """
    # train the model
    model.fit(train_images, train_labels, epochs=10)

    # evaluate the model
    scores = model.evaluate(train_images, train_labels)
    print(f"\n{model.metrics_names[1]}: {scores[1]*100}%")

    # save the model
    model.save('model.h5')

def main():
    """
    This function is the main function.
    """
    # define the model
    model = define_model()

    # train the model
    train_model(model, train_images_norm, labels.values, label_mapping)

    quit()

    # evaluate the model
    scores = model.evaluate(train_images_norm, labels.values)
    print(f"\n{model.metrics_names[1]}: {scores[1]*100}%")

    # predict the test set
    predictions = model.predict(test_images_norm)

    # create the submission file
    sample_submission.prediction = predictions.argmax(axis=1)
    sample_submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()