# The csv with data is in the following format: ####
# label, size_y, size_x, pixel_0, pixel_1...
#
# Since pictures are in different sizes and in csv file every row has to have same amount of columns,
# there is a fix amount of columns (which is equal to size_x*size_y of the biggest picture)
# If a picture is smaller, the last pixels are set to 0

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_pr_dataset():
    df = pd.read_csv('file2.csv', sep=',')

    rawData = np.array(df.values)
    np.random.shuffle(rawData)

    nb_classes = 10

    labels = rawData[:, 0]
    sizes = rawData[:, (1, 2)]
    original_data = rawData[:, 3:]
    data = np.empty(original_data.shape)

    for (i, row) in enumerate(original_data):
        size = sizes[i]
        row = row[0:(size[0] * size[1])]
        # reshaping to rectangle shape
        row = row.reshape(size[1], size[0])
        row = np.rot90(row, 3)
        row = np.flip(row, 1)
        row = row.flatten()
        data[i, 0:row.shape[0]] = row

    print(data[1].shape)

    for i in range(9):
        img = data[i]
        size = sizes[i]
        img = img[0:(size[0] * size[1])]
        # reshaping to rectangle shape
        img = img.reshape(size[0], size[1])
        plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap='gray', interpolation='none')
        plt.title("Class {}".format(labels[i]))

    plt.show()

    # return (x_train, y_train), (x_test, y_test)


get_pr_dataset()
