from keras.layers import Dense, Flatten, Conv2D,  MaxPool2D, Dropout
from common import SEED
from keras.models import Sequential
from keras.optimizers import Adam
from keras import initializers

def create_cnn_keras(lr=0.0004, image_shape=[64,64]):
    model = Sequential()
    initializer = initializers.he_normal(seed=SEED)
    model.add(Conv2D(filters=4, kernel_size=4, strides=1, input_shape=(image_shape[0],image_shape[1],1), padding='same', name='conv_1', activation='relu', kernel_initializer=initializer))
    model.add(MaxPool2D(2, padding='same', strides=1, name='mp_1'))
    filter_size = 8
    for i in range(3):
        model.add(Conv2D(filters=filter_size, kernel_size=4, strides=1, padding='same', name=f'conv_{i+2}', activation='relu', kernel_initializer=initializer))
        model.add(MaxPool2D(2, padding='same', strides=1, name=f'mp_{i+2}'))
        filter_size = filter_size * 2
    model.add(Flatten(name=f'flatten'))
    model.add(Dense(2, activation='softmax', name=f'output_softmax',  kernel_initializer=initializer))
    optimizer = Adam(lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_llr_keras(lr=0.0004, image_shape=[64,64]):
    model = Sequential()
    initializer = initializers.he_normal(seed=SEED)
    model.add(Dense(2, use_bias=False, input_dim=image_shape[0]*image_shape[1], activation='softmax', kernel_initializer=initializer))
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_mlp_keras(lr=0.0004, image_shape=[64,64]):
    n_dim = image_shape[0] * image_shape[1]
    model = Sequential()
    initializer = initializers.he_normal(seed=SEED)
    model.add(Dense(256, input_dim=n_dim, activation='relu', name='fc_1', kernel_initializer=initializer))
    model.add(Dropout(0.3))
    model.add(Dense(256, input_dim=256, activation='relu', name='fc_2', kernel_initializer=initializer))
    model.add(Dropout(0.3))
    model.add(Dense(64, input_dim=256, activation='relu', name='fc_3', kernel_initializer=initializer))
    model.add(Dropout(0.3))
    model.add(Dense(2, input_dim=64, activation='softmax', name='output_softmax', kernel_initializer=initializer))
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

models_dict = {
    "CNN_Keras": create_cnn_keras,
    "LLR_Keras": create_llr_keras,
    "MLP_Keras": create_mlp_keras,
}