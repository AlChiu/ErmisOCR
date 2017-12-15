# Import the necessary packages
import os
import cv2
import argparse
import numpy as np
from PIL import Image


def detector_for_words(image):
    # Read in image with OpenCV
    input_image = cv2.imread(image)

    # Resize (Scale up) the image
    resized_image = cv2.resize(input_image, None, fx=3, fy=3,
                               interpolation=cv2.INTER_CUBIC)

    # Convert the image into a grayscale image.
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Obtain the image dimensions
    resized_height, resized_width = gray_image.shape

    # Blur the image using the bilateral filter
    blur_image = cv2.bilateralFilter(gray_image, 13, 55, 55)

    # Apply Otsu binarization thresholding on the blurred image
    ret, otsu_image = cv2.threshold(blur_image, 0, 255,
                                    cv2.THRESH_BINARY_INV
                                    + cv2.THRESH_OTSU)

    # With the otsu_image, we can start creating bounding boxes of words
    image2, contours, hierarchy = cv2.findContours(otsu_image,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)

    # Let's draw the contours to see what we have
    # contours = contours[-1]
    for contour in contours:
        [x_coord, y_coord, width, height] = cv2.boundingRect(contour)
        cv2.rectangle(resized_image, (x_coord, y_coord),
                      (x_coord+width, y_coord+height),
                      (0, 255, 0), 2)

    # Write this new image into a temp image for testing
    temp_image = "{}.png".format(os.getpid())
    cv2.imwrite(temp_image, resized_image)


if __name__ == "__main__":
    # Build up the argument to bring in an image
    AP = argparse.ArgumentParser()
    AP.add_argument("-i", "--image", required=True, help="path to input image")
    ARGS = vars(AP.parse_args())

    # Draw contours of characters in the image
    detector_for_words(ARGS["image"])
