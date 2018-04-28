"""nist_restructure.py
Used to restructure the NIST SD19 dataset so that it is
usable with Keras' flow from directory functionalilty.
"""
import pathlib
import shutil
import binascii
import argparse

HSF_SET = ['0', '1', '2', '3', '4', '6', '7']


def by_class_extra_removal(directory):
    """
    DESCRIPTION: Use to remove unneccessary files in the
    by_class dataset such as the .mit files and the extra
    images since they are copies. This will leave behind a
    label directory with only a testing and training directory of images.
    INPUT: Path to the by_class dataset root
    OUTPUT: Dataset structure described below:
        by_class
        |
        |-- character label
            |
            |-- testing
                |
                |-- image.png
            |
            |-- training
                |
                |-- image.png
    """
    dataset_path = pathlib.Path(directory)

    # Go into each label path to remove the extra files
    for char in dataset_path.iterdir():
        glo_label = char.parts[-1]
        if glo_label != 'Testing' or glo_label != 'Training':
            # Test dataset
            test_path = char.joinpath("hsf_4")
            test_rename = char.joinpath("testing")

            # Train dataset
            label = char.parts[-1]
            train_path = char.joinpath("train_" + label)
            train_rename = char.joinpath("training")

            # Rename hsf_4 to testing and train_ to training
            if test_path.exists():
                test_path.rename(test_rename)
            if train_path.exists():
                train_path.rename(train_rename)

            # Remove the .mit and extra folders
            for number in HSF_SET:
                # Remove the .mit files
                hsf_mit = char.joinpath("hsf_" + number + ".mit")
                if hsf_mit.exists():
                    hsf_mit.unlink()

                # Remove the hsf_* directories
                hsf_directory = char.joinpath("hsf_" + number)
                if hsf_directory.exists():
                    shutil.rmtree(hsf_directory)


def by_merge_combine(directory):
    """
    DESCRIPTION: Use to merge and create the test and
    training set for each label in the by_merge dataset.
    INPUT: Path to the by_merge dataset root
    OUTPUT: Dataset structure described below:
        by_merge
        |
        |-- character label
            |
            |-- testing
                |
                |-- image.png
            |
            |-- training
                |
                |-- image.png
    """
    dataset_path = pathlib.Path(directory)

    # Go into each label to remove the .mit files and create
    # the test and train files.
    for char in dataset_path.iterdir():
        label = char.parts[-1]
        if label != 'Testing' and label != 'Training':
            # Test dataset
            test_path = char.joinpath("hsf_4")
            test_rename = char.joinpath("testing")
            if test_path.exists():
                test_path.rename(test_rename)

            # Train dataset (Merge hsf_0, 1, 2, 3, 6, 7)
            # Create teh training directory
            train_path = char.joinpath("training")
            if not train_path.exists():
                pathlib.Path(train_path).mkdir(parents=True, exist_ok=True)

            # Go into each hsf directory to move the
            # images in it to the training
            for number in HSF_SET:
                if number != '4':
                    hsf_path = char.joinpath("hsf_" + number)
                    if hsf_path.exists():
                        for old_image in hsf_path.iterdir():
                            old_image_path = pathlib.Path(old_image)
                            image_name = old_image_path.parts[-1]
                            new_image_path = train_path.joinpath(image_name)
                            old_image_path.rename(new_image_path)

                        # Remove the empty hsf folder
                        hsf_path.rmdir()

                # Remove the .mit files
                hsf_mit = char.joinpath("hsf_" + number + ".mit")
                if hsf_mit.exists():
                    hsf_mit.unlink()


def restructure(directory):
    """
    DESCRIPTION: Restructure the provided directory so that it
    follows Keras flow from directory structure.
    INPUT: Dataset directory
    OUTPUT:
        Parent
        |- Training
        |   |- Label
        |       |- Image
        |- Testing
            |- Label
                |- Image
    """
    dataset_path = pathlib.Path(directory)

    test_path = dataset_path.joinpath("Testing")
    if not test_path.exists():
        pathlib.Path(test_path).mkdir(parents=True, exist_ok=True)

    train_path = dataset_path.joinpath("Training")
    if not train_path.exists():
        pathlib.Path(train_path).mkdir(parents=True, exist_ok=True)

    # For each label, we will pull out the training and testing
    # folder and place them in a general training and testing
    # folder. Then we will rename the moved folders to thier labels
    for char in dataset_path.iterdir():
        label = char.parts[-1]
        if label != 'Testing' and label != 'Training':
            # Old training and testing paths
            old_train = char.joinpath("training")
            old_test = char.joinpath("testing")

            # New training and testing paths
            new_train = train_path.joinpath(label)
            new_test = test_path.joinpath(label)

            # Rename/move the old to the new
            if old_train.exists():
                old_train.rename(new_train)
            if old_test.exists():
                old_test.rename(new_test)

            # Remove the old label directory
            if str(char) != str(test_path) and str(char) != str(train_path):
                char.rmdir()


def rename_labels(directory):
    """
    DESCRIPTION: Translate the NIST labels from
    hexadecimal to ascii representation
    INPUT: Dataset path
    OUTPUT: Labels renamed to ascii representation
    """
    dataset_path = pathlib.Path(directory)
    for set_data in dataset_path.iterdir():
        set_data_path = pathlib.Path(set_data)
        for char in set_data_path.iterdir():
            # Limit the label to the first two hex characters for
            # proper translation
            label = char.parts[-1][:2]
            # Translate from hex to ascii
            new_label = (binascii.unhexlify(label)).decode("ascii")
            new_label_path = set_data_path.joinpath(new_label)
            char.rename(new_label_path)


if __name__ == "__main__":
    # Build up the argument to process a directory
    AP = argparse.ArgumentParser()
    AP.add_argument("-d", "--directory",
                    help="Absolute path to dataset directory",
                    required=True)
    AP.add_argument("-s", "--setting",
                    help="class for by_class, merge for by_merge",
                    required=True)
    ARGS = vars(AP.parse_args())

    # First based on the setting, perform the preprocessing of directories
    if ARGS['setting'] == 'class':
        by_class_extra_removal(ARGS['directory'])
    elif ARGS['setting'] == 'merge':
        by_merge_combine(ARGS['directory'])
    else:
        print("Setting is either 'class' or 'merge'.")
        exit()

    # Next, restructure it to follow Keras' input
    restructure(ARGS['directory'])

    # Finally, rename the label folders
    rename_labels(ARGS['directory'])
