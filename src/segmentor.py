"""Segmentation

"""
import cv2


def segment(image, boxes, setting):
    """Used to crop out images based on the bounding boxes"""
    for i, box in enumerate(boxes):
        crop_image = image[box[3]:box[3]+box[5],
                           box[2]:box[2]+box[4]]
        border_crop = cv2.copyMakeBorder(crop_image, 5, 5, 5, 5,
                                         cv2.BORDER_CONSTANT,
                                         value=(255, 255, 255))

        # Write to file name based on the segment setting
        if setting == "word":
            crop_file = "word_" + str(i) + ".png"
        else:
            crop_file = "char_" + str(i) + ".png"

        cv2.imwrite(crop_file, border_crop)
