"""
Configure the file structure of the NIST by_class directory
(RUN ONLY ONCE)
"""
import os
import shutil
import binascii
import argparse


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


if __name__ == "__main__":
    # Build up the argument to bring in an image
    AP = argparse.ArgumentParser()
    AP.add_argument("-d", "--directory", required=True, help="path to dataset")
    ARGS = vars(AP.parse_args())

    NIST_by_class(ARGS["directory"])
