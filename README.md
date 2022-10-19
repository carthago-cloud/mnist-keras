[**version: solved**]

1. `cnn_mnist_keras.py` works with the current version of `cinnaroll`
2. `cnn_mnist_keras_future.py` will hopefully work with an updated version of `cinnaroll`, where model_object is passed to `train_eval()`

# A simple convolutional network in Keras for the MNIST dataset

### Environment setup

Create a virtual environment as follows:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If this does not work and you are on M1 Mac you can try `conda` and `requirements_macos.txt` instead (you can follow instructions given [here](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706)). I have tested this on Linux and macOS with `Python 3.10`.

### Model

To make sure that the model is working, execute a basic routine (creating, training, evaluating) by running:

```
python cnn_mnist_keras.py
```
