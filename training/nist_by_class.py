"""
Configure the file structure of the NIST by_class directory
(RUN ONLY ONCE)
"""
import os
import pathlib
import shutil
import binascii
import argparse
import time


# First we need to process the NIST by_class dataset
def NIST_by_class(directory):
    """
    Go to the NIST by class folder and rename
    and restructure into a usable format
    """
    # First go to the NIST by class directory
    os.chdir(directory)
    dirc = os.listdir()

    for name in dirc:
        test_path = os.path.join(name, "hsf_4")
        train_path = os.path.join(name, "train_" + os.path.splitext(name)[0])

        hsf0_path = os.path.join(name, "hsf_0")
        hsf1_path = os.path.join(name, "hsf_1")
        hsf2_path = os.path.join(name, "hsf_2")
        hsf3_path = os.path.join(name, "hsf_3")
        hsf6_path = os.path.join(name, "hsf_6")
        hsf7_path = os.path.join(name, "hsf_7")

        hsfFile0_path = os.path.join(name, "hsf_0.mit")
        hsfFile1_path = os.path.join(name, "hsf_1.mit")
        hsfFile2_path = os.path.join(name, "hsf_2.mit")
        hsfFile3_path = os.path.join(name, "hsf_3.mit")
        hsfFile4_path = os.path.join(name, "hsf_4.mit")
        hsfFile6_path = os.path.join(name, "hsf_6.mit")
        hsfFile7_path = os.path.join(name, "hsf_7.mit")

        shutil.move(test_path, os.path.join(name, "testing"))
        shutil.move(train_path, os.path.join(name, "training"))

        shutil.rmtree(hsf0_path)
        shutil.rmtree(hsf1_path)
        shutil.rmtree(hsf2_path)
        shutil.rmtree(hsf3_path)
        shutil.rmtree(hsf6_path)
        shutil.rmtree(hsf7_path)

        os.remove(hsfFile0_path)
        os.remove(hsfFile1_path)
        os.remove(hsfFile2_path)
        os.remove(hsfFile3_path)
        os.remove(hsfFile4_path)
        os.remove(hsfFile6_path)
        os.remove(hsfFile7_path)

        b = os.path.splitext(name)[0]
        a = (binascii.unhexlify(b)).decode("ascii")
        shutil.move(name, a)


def split_NIST(directory, division):
    """
    Split the restructured dataset into n subsets
    This function assumes that you already ran the NIST_by_class function
    """
    start = time.time()
    # Change directory and list out the labels
    os.chdir(directory)
    pardir = os.path.abspath(os.pardir)
    dataset = "NIST_SPLIT_224"

    # Create the split dataset parent directory
    split_path = os.path.join(pardir, dataset)
    pathlib.Path(split_path).mkdir(parents=True, exist_ok=True)

    labels = os.listdir()

    for label in labels:
        # Create the label paths
        label_path = os.path.join(directory, label)

        # Go to the label directory
        os.chdir(label_path)

        # List out the testing and training directories
        test_train = os.listdir()

        for image_set in test_train:
            # Create the training and testing directory for each label
            images_path = os.path.join(label_path, image_set)

            # Change to that directory
            os.chdir(images_path)

            # List out the images and count the number per label
            image_files = os.listdir()
            num_files = len(image_files)

            # Divide the count by the number of subsets we want
            subset_count = int(round(num_files / division, 0))

            count = 0
            while count < division:
                if count == division - 1:
                    newlist = image_files[count * subset_count:]

                # Create a a list of subdivided images
                newlist = image_files[count * subset_count:
                                      count * subset_count + subset_count]

                # Create the directory to house the subset of images
                subset = "subset_" + str(count)
                subset_path = os.path.join(split_path,
                                           subset,
                                           image_set,
                                           label)
                pathlib.Path(subset_path).mkdir(parents=True,
                                                exist_ok=True)
                # Move the divided images into the new subsets
                for image in newlist:
                    og_image_path = os.path.join(images_path, image)
                    new_image_path = os.path.join(subset_path, image)
                    shutil.move(og_image_path, new_image_path)
                    print('{} moved to {}'.format(og_image_path,
                                                  new_image_path))

                count += 1
            print('{} has {} images, and each subset will have {} images'
                  .format(images_path, num_files, subset_count))
    print('> Completion time: {}'.format(time.time() - start))
    return split_path, division


if __name__ == "__main__":
    # Build up the argument to bring in an image
    AP = argparse.ArgumentParser()
    AP.add_argument("-d", "--directory", required=True, help="path to dataset")
    ARGS = vars(AP.parse_args())

    NIST_by_class(ARGS["directory"])
