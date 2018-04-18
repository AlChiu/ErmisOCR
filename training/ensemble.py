"""
Used to ensemble a number of different trained models
into a single model.
"""
from keras.layers import Input, average
from keras.models import load_model, Model


def ensemble(models, model_input):
    """
    Function to combine multiple models into one.
    Input layer is the same since the images will
    have to have the same dimensions.
    Output layer will average the results and classify.
    """
    # Collect the outputs of the models
    output = [model(model_input) for model in models]

    # Average the outputs
    avg = average(output)

    # Build the one model
    model_ens = Model(inputs=model_input, outputs=avg,
                      name='ensemble')

    return model_ens


if __name__ == "__main__":
    MODELS = []

    # Load the three models
    TEMP_MODEL_1 = load_model("model62_lenet_rmsprop.hdf5")
    TEMP_MODEL_2 = load_model("model62_lenet2_rmsprop.hdf5")
    TEMP_MODEL_3 = load_model("model62_lenet3_rmsprop.hdf5")

    # Append to list
    MODELS.append(TEMP_MODEL_1)
    MODELS.append(TEMP_MODEL_2)
    MODELS.append(TEMP_MODEL_3)

    # Create the ensemble input layer
    INPUT = Input(shape=MODELS[0].input_shape[1:])

    # Ensemble the models
    ENS_MODEL = ensemble(MODELS, INPUT)
    ENS_MODEL.summary()
    ENS_MODEL.save("model62_ensemble.hdf5")
