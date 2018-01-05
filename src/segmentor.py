"""Segmentation

"""
import os
import cv2


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
        cv2.imwrite(crop_file, border_crop)
