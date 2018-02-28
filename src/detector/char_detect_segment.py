"""char_detect_segment
This module contains the necessary functions to detect characters,
merge bouding boxes of the characters, and crop them out.
"""
import os
import collections
import glob
import re
import pathlib
from operator import itemgetter
import cv2
import numpy as np

NUMBERS = re.compile(r'(\d+)')
IMAGEBOXES = collections.namedtuple('Image_Boxes', ['Name',
                                                    'Image',
                                                    'Boxes'])
PATH = "/home/alexander/Desktop/projects/ErmisOCR/src/classifier/segmented/"


def numerical_sort(value):
    """Used to sort file names numberically instead of alphabetically"""
    parts = NUMBERS.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def sort_contours(contours, setting):
    """Sort the contours by x and y coordinates
     and calculate the central coordinates"""
    bound_box = []
    for contour in contours:
        [x_coord, y_coord, width, height] = cv2.boundingRect(contour)
        center_x = x_coord + width / 2
        center_y = y_coord + height / 2
        single_box = [center_x, center_y, x_coord, y_coord, width, height]
        bound_box.append(single_box)

    if setting == "word":
        # Need to sort the bounding bound_box by line first
        # Then sort by characters in the line
        # First, sort the contours by the y-coordinate
        bound_box.sort(key=itemgetter(3))
        # Set the initial line threshold to the bottom of the first character
        line_theshold = bound_box[0][3] + bound_box[0][5] - 1
        # If any top y coordinate is less than the line threshold,
        # it is a new line
        l_start = 0
        for i, char_bound_box in enumerate(bound_box):
            if char_bound_box[3] > line_theshold:
                # Sort the line by the x-coordinate
                bound_box[l_start:i] = sorted(bound_box[l_start:i],
                                              key=itemgetter(2))
                l_start = i
                line_theshold = max(char_bound_box[3]+char_bound_box[5]-1,
                                    line_theshold)
        # Sort the last line
        bound_box[l_start:] = sorted(bound_box[l_start:], key=itemgetter(2))
    else:
        # Just sort by the x-coordinate
        bound_box.sort(key=itemgetter(2))

    return bound_box


def dis(start, end):
    """Calculate the Euclidean distance in one direction"""
    return abs(start - end)


def create_word_mask(boxes, height, width):
    """Create a mask for our words"""
    # Create a blank mask
    mask = np.zeros((height, width), dtype="uint8")

    # Using the bounding boxes, we will draw white boxes on the mask
    for mask_box in boxes:
        cv2.rectangle(mask, (mask_box[0], mask_box[1]),
                      (mask_box[0]+mask_box[2], mask_box[1]+mask_box[3]),
                      255, -1)

    # With this mask, we will have to unidirectionally dilate so that
    # characters within the word are connected.
    # Kernel is used to dilate in the horizontal direction
    #
    # First, we need to determine the size of the kernel
    # We need two row matrices (one is zeros and other is ones).
    # The size of the zero kernel is relative to the width of the image
    # while the one kernel is just +1 of the zero kernel.
    k_1_width = int(.005 * width)
    k_2_width = k_1_width + 1
    kernel_1 = np.zeros((1, k_1_width), dtype="uint8")
    kernel_2 = np.ones((1, k_2_width), dtype="uint8")

    # We the combine the two kernels into one large row kernel for dilation
    kernel = np.append(kernel_1, kernel_2)
    kernel.shape = (1, k_1_width + k_2_width)

    # With the new kernel, we dilate the mask image.
    mask = cv2.dilate(mask, kernel, iterations=1)

    # With this new mask image, we find new contours and bounding boxes.
    _, contours, _ = cv2.findContours(mask,
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    mask_contours = sort_contours(contours, "word")

    return mask_contours


def merge_boxes(contours, height, width, setting):
    """Merge the bounding boxes

    Input -- list of character bounding boxes, image height, and image width
    Output -- list of new character bounding boxes for special characters
    """
    # New list of bounding boxes
    merged = []

    # Produce a sorted list of bounding box coordinates with their centers
    current_list = sort_contours(contours, setting)

    # Thresholds for checking multi-component characters
    if setting == "word":
        h1 = .105 * height
        h2 = .02 * height
        w1 = .01 * width
    else:
        h1 = .75 * height
        h2 = .2 * height
        w1 = .05 * width

    # Need to merge boxes of multi-component
    # characters i, j, !, ?, :, ;, =, %, and ".
    empty = [0, 0, 0, 0, 0, 0]
    prev_list = [empty] + current_list
    aft_list = current_list[1:] + [empty]

    for bef, cur, aft in zip(prev_list, current_list, aft_list):
        # First check the current and previous box
        if(dis(cur[0], bef[0]) <= w1) and (h2 < dis(cur[1], bef[1]) <= h1):
            # Check three boxes for the % character
            if(dis(cur[0], aft[0]) <= w1) and (h2 < dis(cur[1], aft[1] <= h1)):
                # We have a potential % character
                new_x = min(bef[2], cur[2], aft[2])
                new_y = min(bef[3], cur[3], aft[3])
                new_w = max(cur[4], aft[2]+aft[4]-new_x, cur[2]+cur[4]-new_x)
                new_h = max(cur[5], aft[3]+aft[5]-new_y, cur[3]+cur[5]-new_y)
            else:
                new_x = min(bef[2], cur[2])
                new_y = min(bef[3], cur[3])
                # Check for the quotation mark
                if dis(cur[1], bef[1]) <= h2:
                    new_w = bef[4] + cur[4]
                    new_h = max(bef[5], cur[5])
                else:
                    # The next set of characters are the vertical
                    # two-component characters
                    new_w = max(bef[4], cur[2]+cur[4]-bef[2])
                    new_h = max(cur[3]+cur[5]-bef[3], bef[3]+bef[5]-cur[3])
        # Stand alone character
        else:
            new_x = cur[2]
            new_y = cur[3]
            new_w = cur[4]
            new_h = cur[5]
        merged.append([new_x, new_y, new_w, new_h])

    return merged


def detector_for_words(full_image):
    """Detect image for words

    Input -- image that needs translation
    Ouput -- list of bounding box coordinates for the words
    """
    filename = os.path.splitext(os.path.basename(full_image))[0]

    # Read in image and resize (Scale up) the image
    resized_image = cv2.resize(cv2.imread(full_image), None, fx=3, fy=3,
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
    boxes = merge_boxes(contours, resized_height, resized_width, "word")
    mask_boxes = create_word_mask(boxes, resized_height, resized_width)
    result = IMAGEBOXES(filename, resized_image, mask_boxes)

    return result


def detector_for_characters(word_image):
    """Detect characters from word images"""
    filename = os.path.splitext(os.path.basename(word_image))[0]
    word = cv2.imread(word_image)

    # Convert to grayscale image
    gray_word = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)

    # Obtain the image dimensions
    height, width = gray_word.shape

    # Blur the image
    gray_blur = cv2.bilateralFilter(gray_word, 13, 55, 55)

    # Otsu binarization
    _, gray_otsu = cv2.threshold(gray_blur, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Contours
    _, char_contours, _ = cv2.findContours(gray_otsu,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Sort and merge the contours, get the list of bounding boxes
    characters = merge_boxes(char_contours, height, width, "char")
    result = IMAGEBOXES(filename, word_image, characters)

    return result


def segment(filename, image, boxes, setting):
    """Used to crop out images based on the bounding boxes"""
    for i, box in enumerate(boxes):
        # Write to file name based on the segment setting
        if setting == "word":
            crop_img = image[box[3]:box[3]+box[5],
                             box[2]:box[2]+box[4]]
            crop_file = "word_" + str(i) + ".png"

        else:
            crop_img = image[box[1]:box[1]+box[3],
                             box[0]:box[0]+box[2]]
            path = os.path.splitext(os.path.basename(filename))[0]
            crop_file = "char_" + str(i) + "_" + path + ".png"

        border_crop = cv2.copyMakeBorder(crop_img, 5, 5, 5, 5,
                                         cv2.BORDER_CONSTANT,
                                         value=(255, 255, 255))
        pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
        image_path = PATH + crop_file
        cv2.imwrite(image_path, border_crop)


def detect_segment(image):
    """
    Perform the character detection and segment them out
    """
    # Draw contours of characters in the image
    word_boxes = detector_for_words(image)
    segment(word_boxes.Name,
            word_boxes.Image,
            word_boxes.Boxes,
            "word")

    for word in sorted(glob.iglob(PATH + 'word_*.png'), key=numerical_sort):
        char_boxes = detector_for_characters(word)
        w_image = cv2.imread(word)

        # Segment / Crop the characters out
        segment(char_boxes.Name,
                w_image,
                char_boxes.Boxes,
                "char")

        os.remove(word)
