# Ref: https://keras.io/examples/vision/mnist_convnet/

import json
from pathlib import Path
import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
import cinnaroll


NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)


def load_data(num_classes, limit):
    # Load the data and split it between train and test sets
    data_array = keras.datasets.mnist.load_data()

    if limit is not None:
        tmp = data_array

        data_array = [[None] * 2, [None] * 2]
        for j in range(2):
            for k in range(2):
                data_array[j][k] = tmp[j][k][:limit]

    all_data = {
        "test": {"X": data_array[1][0], "Y": data_array[1][1]},
        "train": {"X": data_array[0][0], "Y": data_array[0][1]},
    }

    print("\nDataset info:")

    for key, val in all_data.items():
        print(f"Number of samples in {key} set: {val['X'].shape[0]}")
        # the range of each pixel should be a float in [0, 1] interval
        val["X"] = val["X"].astype("float32") / 255
        # every image should have shape (28, 28, 1)
        val["X"] = np.expand_dims(val["X"], -1)
        # convert numerical labels to one-hot encodings
        val["Y"] = keras.utils.to_categorical(val["Y"], num_classes)

    return all_data


def construct_model(num_classes, input_shape):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # print model summary
    print("\nModel info:")
    model.summary()

    # compile model
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )

    return model


def train_model(model, all_data, batch_size, epochs):
    model.fit(
        all_data["train"]["X"],
        all_data["train"]["Y"],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(all_data["test"]["X"], all_data["test"]["Y"]),
    )


def evaluate_model(model, data):
    score = model.evaluate(data["X"], data["Y"], verbose=0)
    print(f"Test loss: {score[0]:.3f}")
    print(f"Test accuracy: {score[1]:.3f}")


def preprocess_image(input_data):
    img = PIL.Image.open(input_data)
    img_processed = img.convert("L").resize(INPUT_SHAPE[:2])
    img_array = np.array(img_processed).reshape((1, ) + INPUT_SHAPE)
    return img_array


def make_prediction(x):
    return int(np.argmax(x))


# def infer(model_object, input_data):
#     img = PIL.Image.open(input_data)
#     img_processed = img.convert("L").resize(INPUT_SHAPE[:2])
#     img_array = np.array(img_processed).reshape((1, ) + INPUT_SHAPE)
#     output = model_object.predict(img_array)
#     return json.dumps(int(np.argmax(output)))


def train(model_object, training_data, epochs):
    model_object.fit(
        training_data["X"],
        training_data["Y"],
        batch_size=128,
        epochs=epochs
    )


def generate_and_test_model_config():
    all_data = load_data(num_classes=NUM_CLASSES, limit=100)
    model_object = construct_model(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE)
    train(model_object, all_data["train"], 50)

    # generate sample input to the model and test it
    sample_input = tf.expand_dims(all_data["test"]["X"][0, :, :, :], 0)
    model_object.predict(sample_input)

    # generate sample input to the infer function and test it
    img_file = Path(__file__).parent / "test_image.png"
    infer(model_object, img_file)

    # evaluate model to compute accuracy
    accuracy = model_object.evaluate(all_data["test"]["X"], all_data["test"]["Y"], verbose=0)[1]

    model_config = {
        "project_id": "2eb823ea",
        "model_object": model_object,
        "model_input_sample": sample_input,
        "infer_func": infer,
        "infer_func_input_format": "img",
        "infer_func_output_format": "json",
        "infer_func_input_sample": str(img_file),
        "train_func": train,
        "metrics":
            {
                'dataset': "MNIST",
                'accuracy': round(accuracy, 3),
            }
    }

    print(model_config)


class MyRolloutConfig(cinnaroll.RolloutConfig):
    @staticmethod
    def train_eval(): # training and evaluation with metric extraction
        all_data = load_data(num_classes=NUM_CLASSES, limit=100)
        X = all_data["train"]["X"]
        Y = all_data["train"]["Y"]

        model.fit(X, Y, epochs=5)
    @staticmethod
    def infer(model_object, input_data): # input -> processing -> inference -> output
        img_array = preprocess_image(input_data)
        out = model_object.predict(img_array)
        return json.dumps({"output": make_prediction(out)})



# define the number of classes and expected input shape
all_data = load_data(num_classes=NUM_CLASSES, limit=100)
model = construct_model(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE)

model_input_sample = all_data["test"]["X"][0, :, :, :].reshape(1, 28, 28, 1)
infer_func_input_sample = os.getcwd()+"/test_image.png"

myRolloutConfig = MyRolloutConfig(
    project_id="5zVm00n97",  # project's unique identifier
    model_object=model,
    model_input_sample=model_input_sample,  # sample you can pass to model object's predict function
    infer_func_input_format="img",  # "json", "img" or "file"
    infer_func_output_format="json",  # "json" or "img" currently supported
    infer_func_input_sample=infer_func_input_sample,  # note - for file or img just pass file path
)

cinnaroll.rollout(myRolloutConfig)
