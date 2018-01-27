import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

max_height = 600
max_width = 800


def separate_handwritten_digits(im_file='digits.jpeg', imshow=False):
    im_gray = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
    if im_gray is None:
        print 'Cannot find image "{}", did you forget to specify the directory?'.format(im_file)
        return []
    height, width = im_gray.shape[:2]

    # Only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        im_gray = cv2.resize(im_gray, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    if imshow:
        cv2.imshow('Original image', im_gray)

    (thresh, im_bw) = cv2.threshold(im_gray, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img = cv2.bitwise_not(im_bw)

    # Remove noise
    # img = cv2.erode(img, np.ones((2, 2)))
    # img = cv2.dilate(img, np.ones((2, 2)))

    if imshow:
        cv2.imshow('Thresholded denoised image', img)

    # Join separate small objects so bounding boxes can be found
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    img_joined = cv2.dilate(img, kernel)
    if imshow:
        cv2.imshow('Joined image (for finding bounding boxes)', img_joined)

    # Compute contours so bounding boxes can be found, only keep outer contours (not the o inside O) with RETR_EXTERNAL
    contours, _ = cv2.findContours(img_joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    # Prepare color image to draw bounding boxes on
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # img_color[img_color == 255] = 1

    count = 0
    cells = []
    labels = [4,2,3,6,5,1,0,9,7,8,3,2,4,0,1,9,5,8,7,6,2,9,1,3,4,0,8,7,5,6,2,1,9,4,3,8,7,0,5,6,1,2,9,7,3,6,4,0,8,5,1,9,7,0,6,2,5,4,3,8,9,2,7,1,5,8,6,4,3,0,4,5,3,2,1,0,9,7,6,8,0,2,1,4,3,9,7,5,8,6]
    data = []
    for i in range(len(contours)):
        c = contours[i]
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)

        cell = img[y:y + h, x:x + w]
        cells.append(cell)
        count += 1
        if imshow:
            cv2.rectangle(img_color, (x-8, y-8), (x + w+8, y + h + 8), (0, 0, 255), 2)
        letter = img_color[y:y+h,x:x+w]
        letter = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
        # plt.imshow(letter, cmap='gray')
        # plt.show()
        data.append([labels[i], letter.shape[1], letter.shape[0]] + list(letter.flatten()))


    fil = open("self-digits.csv", 'wb')
    for r in data:
        fil.write(str(r).strip("[]"))

    fil.close()

    # np.savetxt("self-digits.csv", data , delimiter=",")



    if imshow:
        cv2.imshow('Digit bounding boxes', img_color)
        cv2.imwrite("boundingbox.png", img_color)
        cv2.waitKey(0)
    return cells


if __name__ == "__main__":
    separate_handwritten_digits(imshow=True)
