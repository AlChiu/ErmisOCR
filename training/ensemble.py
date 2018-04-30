"""ensemnble.py
Used to ensemble a number of different trained models
into a single model.
"""
import argparse
from keras.layers import Input, average
from keras.models import load_model, Model
import train_classifier as tc


def ensemble(models, model_input):
    """
    DESCRIPTION: Combine multiple models into one.
    Input layer is the same since the images will
    have to have the same dimensions. Output layer
    will average the results and classify.
    INPUT: List of models that we want to ensemble,
    Dimensions of the input image in tuple format
    OUTPUT: Compiled Keras model of our ensemble.
    """
    # Collect the outputs of the models
    output = [model(model_input) for model in models]

    # Averaging Layer
    avg = average(output)

    # Build the ensembled model
    model_ens = Model(inputs=model_input, outputs=avg,
                      name='ensemble')

    model_ens.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

    return model_ens


def evaluate_error(model, test_gen, no_test_sam):
    """
    DESCRIPTION: Evaluate the network performance
    INPUT: The trained model, the test set generator,
    number of testing images
    OUTPUT: List of statistics about the performance of the model
    """
    stats = model.evaluate_generator(test_gen, no_test_sam // 256)
    return stats


if __name__ == "__main__":
    # Build arguments
    AP = argparse.ArgumentParser()
    AP.add_argument("-d", "--data_path", help="path to dataset", required=True)
    ARGS = vars(AP.parse_args())

    MODELS = []
    PATH = ARGS['data_path']

    # Load the three models
    TEMP_MODEL_1 = load_model("src/classifier/model47_simple1.hdf5")
    TEMP_MODEL_2 = load_model("src/classifier/model47_simple2.hdf5")
    TEMP_MODEL_3 = load_model("src/classifier/model47_simple3.hdf5")

    # Append to list
    MODELS.append(TEMP_MODEL_1)
    MODELS.append(TEMP_MODEL_2)
    MODELS.append(TEMP_MODEL_3)

    # Create the ensemble input layer
    INPUT = Input(shape=MODELS[0].input_shape[1:])

    # Ensemble the models
    ENS_MODEL = ensemble(MODELS, INPUT)
    ENS_MODEL.save("src/classifier/model47_ensemble.hdf5")
    ENSEMBLE = load_model("src/classifier/model47_ensemble.hdf5")
    MODELS.append(ENSEMBLE)

    _, TEST_GEN, _, NB_TEST_SAM, _ = tc.create_feed_data(PATH, 32, 32)
    # Evaluate all of the models
    for model in MODELS:
        score = evaluate_error(model, TEST_GEN, NB_TEST_SAM)
        model.summary()
        print("Loss: ", score[0], "Accuracy: ", score[1])
