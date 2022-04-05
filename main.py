# import statements
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

LOG = False

# training of dataset
labels = pd.read_csv('data/train_labels.csv')  # read labels
train_images = np.load('data/train_images.npy') # load train images

# mapping
label_mapping = pd.read_csv('data/label_int_to_str_mapping.csv')  # read label mapping

# testing part of dataset
sample_submission = pd.read_csv('data/sample_submission.csv') # read sample_submission
test_images = np.load('data/test_images.npy') # load test images

if (LOG):
    print(f"""
    {len(labels)= }
    {len(label_mapping)= }
    {len(sample_submission)= }
    {len(test_images)= }
    {len(train_images)= }
    """)
    breakpoint()

# plot the first 9 images
f, axarr = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        axarr[i, j].imshow(train_images[i*3+j].reshape(28, 28))
        axarr[i, j].set_title(f"Train Image {i*3+j}")
        axarr[i, j].axis('off')

plt.show()