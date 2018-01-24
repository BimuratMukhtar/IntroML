from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

#-------------------------------------------------------------------------------
# Part 1 - Fully Connected Neural Network
#-------------------------------------------------------------------------------

def build_model_fc(input_shape, num_classes):
    """Builds a fully connected neural network.

    Layers:
        - Dense, 512 neurons, relu activation
        - Dropout, 0.2 probability
        - Dense, 512 neurons, relu activation
        - Dropout, 0.2 probability
        - Dense, num_classes neurons, softmax activation

    Arguments:
        input_shape(tuple): The shape of the input to the network.
        num_classes(int): The number of classes being predicted
                          (i.e. the size of the output of the network).

    Returns:
        A fully connected keras model.
    """
    model = Sequential()
    model.add(Dense(input_shape=input_shape, units=512, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=num_classes, activation="softmax"))

    return model

#-------------------------------------------------------------------------------
# Part 2 - Convolutional Neural Network
#-------------------------------------------------------------------------------

def build_model_conv(input_shape, num_classes):
    """Builds a convolutional neural network.

    Layers:
        - Conv2D, 32 3x3 filters, relu activation
        - Conv2D, 64 3x3 filters, relu activation
        - MaxPooling2D, 2x2 pool size
        - Dropout, 0.25 probability
        - Flatten
        - Dense, 128 neurons, relu activation
        - Dropout, 0.5 probability
        - Dense, num_classes neurons, softmax activation

    Arguments:
        input_shape(tuple): The shape of the input to the network.
        num_classes(int): The number of classes being predicted
                          (i.e. the size of the output of the network).

    Returns:
        A convolutional keras model.
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, activation="softmax"))

    return model
