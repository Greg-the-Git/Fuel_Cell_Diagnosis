from keras import models, layers


def CNN():
    """
    Create a Convolutional Neural Network (CNN) model.

    Returns:
    model (keras.Sequential): CNN model.
    """
    model = models.Sequential()
    model.add(layers.Reshape((40001, 1), input_shape=(40001,)))
    model.add(layers.Conv1D(64, kernel_size=100, strides=10))
    model.add(layers.MaxPooling1D(pool_size=3, strides=2))
    model.add(layers.Conv1D(32, kernel_size=5, strides=10))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(3, activation="softmax"))

    return model


def DNN():
    """
    Create a Deep Neural Network (DNN) model.

    Returns:
    model (keras.Sequential): DNN model.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=40001))
    model.add(layers.Dense(units=512, activation="relu"))
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(units=64, activation="relu"))
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(3))
    model.add(layers.Activation("softmax"))

    return model
