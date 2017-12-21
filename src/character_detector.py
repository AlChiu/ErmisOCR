"""Character Detection

This module contains the necessary scripts to detect characters and merge
bouding boxes of the characters.

"""
import argparse
from operator import itemgetter
import cv2
import numpy as np


def create_word_mask(boxes, height, width):
    """Create a mask for our words"""
    # Create a blank mask
    mask = np.zeros((height, width), dtype="uint8")

    # Using the bounding boxes, we will draw white boxes on the mask
    for box in boxes:
        cv2.rectangle(mask, (box[0], box[1]),
                      (box[0]+box[2], box[1]+box[3]),
                      (255, 255, 255), -1)

    # With this mask, we will have to unidirectionally dilate so that
    # characters within the word are connected.
    return mask


def sort_contours(contours):
    """Sort the contours by x and y coordinates
     and calculate the central coordinates"""
    boxes = []
    for contour in contours:
        [x_coord, y_coord, width, height] = cv2.boundingRect(contour)
        center_x = x_coord + width / 2
        center_y = y_coord + height / 2
        box = [center_x, center_y, x_coord, y_coord, width, height]
        boxes.append(box)

    # Need to sort this list by x and y-coordinates
    boxes = sorted(boxes, key=itemgetter(2, 3))

    return boxes


def dis(x, y):
    """Calculate the Euclidean distance in one direction"""
    return abs(x - y)


def merge_boxes(contours, height, width):
    """Merge the bounding boxes

    Input -- list of character bounding boxes, image height, and image width
    Output -- list of new character bounding boxes for special characters
    """
    # Thresholds for checking multi-component characters
    h1 = .12 * height
    h2 = .05 * height
    w1 = .05 * width

    # New list of bounding boxes
    merged = []

    # Produce a sorted list of bounding box coordinates with their centers
    box = sort_contours(contours)

    # Need to merge boxes of multi-component
    # characters i, j, !, ?, :, ;, =, %, and ".
    empty = (0, 0, 0, 0, 0, 0)
    prev_list = [empty] + box
    aft_list = box + [empty]

    for bef, cur, aft in zip(prev_list, box, aft_list):
        # First check the current and previous box
        if(dis(cur[0], bef[0]) <= w1) and (h2 < dis(cur[1], bef[1]) <= h1):
            # Check three boxes for the % character
            if(dis(cur[0], aft[0]) <= w1) and (h2 < dis(cur[1], aft[1] <= h1)):
                # We have a potential % character
                new_x = min(bef[2], cur[2], aft[2])
                new_y = min(bef[3], cur[3], aft[3])
                new_w = max(bef[4], cur[4], aft[4])
                new_h = max(bef[5], cur[5], aft[5])
            else:
                new_x = min(bef[2], cur[2])
                new_y = min(bef[3], cur[3])
                # Check for the quotation mark
                if(dis(cur[1], bef[1]) <= h2):
                    new_w = bef[4] + cur[4]
                    new_h = max(bef[5], cur[5])
                else:
                    # The next set of characters are the vertical
                    # two-component characters
                    # NEED TO ASSOCIATE THE MIN_Y HEIGHT
                    new_w = max(bef[4], cur[4])
                    new_h = bef[5] + cur[5] + (max(bef[3], cur[3])
                                               - (new_y + min(bef[5], cur[5])))
        # Stand alone character
        else:
            new_x = cur[2]
            new_y = cur[3]
            new_w = cur[4]
            new_h = cur[5]
        merged.append([new_x, new_y, new_w, new_h])

    return merged


def detector_for_words(image):
    """Detect image for words

    Input -- image that needs translation
    Ouput -- list of bounding box coordinates for the words
    """
    # Read in image and resize (Scale up) the image
    resized_image = cv2.resize(cv2.imread(image), None, fx=3, fy=3,
                               interpolation=cv2.INTER_CUBIC)

    # Convert the image into a grayscale image.
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Obtain the image dimensions
    resized_height, resized_width = gray_image.shape

    # Blur the image using the bilateral filter
    blur_image = cv2.bilateralFilter(gray_image, 13, 55, 55)

    # Apply Otsu binarization thresholding on the blurred image
    _, otsu_image = cv2.threshold(blur_image, 0, 255,
                                  cv2.THRESH_BINARY_INV
                                  + cv2.THRESH_OTSU)

    # With the otsu_image, we can start creating bounding boxes of words
    _, contours, _ = cv2.findContours(otsu_image,
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

    # Let's draw the contours to see what we have
    boxes = merge_boxes(contours, resized_height, resized_width)
    mask = create_word_mask(boxes, resized_height, resized_width)
    masked_image = cv2.bitwise_and(resized_image, resized_image, mask=mask)

    # Show this new image for testing
    cv2.imshow("bounding_boxes", masked_image)
    cv2.waitKey(0)

    return boxes


if __name__ == "__main__":
    # Build up the argument to bring in an image
    AP = argparse.ArgumentParser()
    AP.add_argument("-i", "--image", required=True, help="path to input image")
    ARGS = vars(AP.parse_args())

    # Draw contours of characters in the image
    detector_for_words(ARGS["image"])
