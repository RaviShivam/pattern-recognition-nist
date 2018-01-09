# The csv with data is in the following format: ####
# label, size_y, size_x, pixel_0, pixel_1...
#
# Since pictures are in different sizes and in csv file every row has to have same amount of columns,
# there is a fix amount of columns (which is equal to size_x*size_y of the biggest picture)
# If a picture is smaller, the last pixels are set to 0

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DESIRED_SQUARE_SIZE = 30


def __get_raw_pr_dataset():
    df = pd.read_csv('file2.csv', sep=',')

    rawData = np.array(df.values)
    np.random.shuffle(rawData)

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

    # for i in range(9):
    #     img = data[i]
    #     size = sizes[i]
    #     img = img[0:(size[0] * size[1])]
    #     # reshaping to rectangle shape
    #     img = img.reshape(size[0], size[1])
    #     plt.subplot(3, 3, i + 1)
    #     plt.imshow(img, cmap='gray', interpolation='none')
    #     plt.title("Class {}".format(labels[i]))
    #
    # plt.show()

    return labels, sizes, data


def __get_squared_dataset(labels, sizes, data):
    new_data = np.zeros((data.shape[0], DESIRED_SQUARE_SIZE ** 2))

    for (i, flat_img) in enumerate(data):
        size = sizes[i]
        img = flat_img[0:(size[0] * size[1])]
        img = img.reshape(size[0], size[1])

        # plt.subplot(1, 2, 1)
        # plt.imshow(img, cmap='gray', interpolation='none')

        h, w = img.shape[:2]
        sh, sw = (max(size), max(size))

        # interpolation method
        if h > sh or w > sw:  # shrinking image
            interp = cv2.INTER_AREA
        else:  # stretching image
            interp = cv2.INTER_CUBIC

        # aspect ratio of image
        aspect = w / h

        pad_color = 0
        # compute scaling and pad sizing
        if aspect > 1:  # horizontal image
            new_w = sw
            new_h = np.round(new_w / aspect).astype(int)
            pad_vert = (sh - new_h) / 2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        elif aspect < 1:  # vertical image
            new_h = sh
            new_w = np.round(new_h * aspect).astype(int)
            pad_horz = (sw - new_w) / 2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        else:  # square image
            new_h, new_w = sh, sw
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

        # scale and pad
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                        borderType=cv2.BORDER_CONSTANT, value=pad_color)
        scaled_img = cv2.resize(scaled_img, (DESIRED_SQUARE_SIZE, DESIRED_SQUARE_SIZE), interpolation=cv2.INTER_NEAREST)

        # plt.subplot(1, 2, 2)
        # plt.imshow(scaled_img, cmap='gray', interpolation='none')
        # plt.show()

        scaled_img = scaled_img.flatten()
        new_data[i] = scaled_img

    return labels, new_data


(labels, sizes, data) = __get_raw_pr_dataset()

(labels, data) = __get_squared_dataset(labels, sizes, data)
plt.imshow(data[1].reshape(DESIRED_SQUARE_SIZE, DESIRED_SQUARE_SIZE), cmap='gray', interpolation='none')
plt.show()
