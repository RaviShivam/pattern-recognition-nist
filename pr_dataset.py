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
PIXEL_THRESHOLD = 0.6


def plot_image(data_to_display, rows, cells, index, size=DESIRED_SQUARE_SIZE):
    img = data_to_display
    img = img.reshape(size, size)
    plt.subplot(rows, cells, index)
    plt.imshow(img, cmap='gray', interpolation='none')


def __get_raw_pr_dataset():
    df = pd.read_csv('file2.csv', sep=',')

    raw_data = np.array(df.values)
    np.random.shuffle(raw_data)

    labels = raw_data[:, 0]
    sizes = raw_data[:, (1, 2)]
    original_data = raw_data[:, 3:]
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


def __get_squared_dataset(sizes, data):
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

    return new_data


def __deskew(data):
    skew_data = np.zeros((data.shape[0], DESIRED_SQUARE_SIZE * DESIRED_SQUARE_SIZE))

    for (i, img) in enumerate(data):
        img = img[0:(DESIRED_SQUARE_SIZE * DESIRED_SQUARE_SIZE)]
        img = img.reshape(DESIRED_SQUARE_SIZE, DESIRED_SQUARE_SIZE)

        moments = cv2.moments(img)
        if abs(moments['mu02']) < 1e-2:
            return img.copy()
        skew = moments['mu11'] / moments['mu02']
        moments = np.float32([[1, skew, -0.5 * DESIRED_SQUARE_SIZE * skew], [0, 1, 0]])
        img = cv2.warpAffine(img, moments, (DESIRED_SQUARE_SIZE, DESIRED_SQUARE_SIZE),
                             flags=cv2.WARP_INVERSE_MAP)

        img = img.flatten()
        skew_data[i, 0:img.shape[0]] = img

    return skew_data


def __sharpen_image(data):
    sharp_images = np.zeros(data.shape)

    for (i, xxx) in enumerate(data):
        img = data[i]

        img = img.reshape(DESIRED_SQUARE_SIZE, DESIRED_SQUARE_SIZE)
        # plot_image(img, 1, 4, 1)
        tmp_size = DESIRED_SQUARE_SIZE * 6
        img = cv2.resize(img, (tmp_size, tmp_size))
        # img = cv2.blur(img, (5, 5))
        img = cv2.GaussianBlur(img, (25, 25), 0)
        # plot_image(img, 1, 4, 2, size=tmp_size)
        img = cv2.resize(img, (DESIRED_SQUARE_SIZE, DESIRED_SQUARE_SIZE))

        # plot_image(img, 1, 4, 3)

        img = img.flatten()
        for (j, row) in enumerate(img):
            img[j] = 1 if img[j] > PIXEL_THRESHOLD else 0

        # plot_image(img, 1, 4, 4)
        # plt.show()

        sharp_images[i] = img
    # plot_image(sharp_images[1], 1, 1, 1)
    # plt.show()
    return sharp_images


def get_preprocessed_dataset(plot=False):
    (labels, sizes, data) = __get_raw_pr_dataset()

    data = __get_squared_dataset(sizes, data)

    if plot:
        plot_image(data[0], 1, 3, 1)

    data = __deskew(data)

    if plot:
        plot_image(data[0], 1, 3, 2)

    data = __sharpen_image(data)

    if plot:
        plot_image(data[0], 1, 3, 3)
        plt.show()

    size_tuple = (DESIRED_SQUARE_SIZE, DESIRED_SQUARE_SIZE)
    return labels, data, size_tuple


def save_dataset_to_csv():
    labels, data, size_tuple = get_preprocessed_dataset()
    full_data = np.zeros((labels.shape[0], 1 + data.shape[1]), np.int32)
    full_data[:, 0] = labels[:]
    full_data[:, 1:] = data.astype(int)
    np.savetxt("data/preprocessed_30.csv", full_data, delimiter=",", fmt='%d')
    print('saved to csv')


# get_preprocessed_dataset(True)
save_dataset_to_csv()
