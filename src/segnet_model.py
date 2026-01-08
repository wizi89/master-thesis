from keras.layers.noise import GaussianNoise
import keras.models as models
from keras.layers.core import Activation, Reshape, Permute, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

K.set_image_dim_ordering('th')

def autoencoder(img_size, klassen, num_channels):
    
    nb_classes = len(klassen)

    autoencoder = models.Sequential()
    
    autoencoder.encoding_layers = create_encoding_layers(img_size, num_channels)
    autoencoder.decoding_layers = create_decoding_layers()
    for l in autoencoder.encoding_layers:
        autoencoder.add(l)
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)
    
    autoencoder.add(Convolution2D(nb_classes, 1, 1, border_mode='valid',))
    autoencoder.add(Reshape((nb_classes,img_size[0]*img_size[1]), input_shape=(nb_classes,img_size[0],img_size[1])))
    autoencoder.add(Permute((2, 1)))
    autoencoder.add(Flatten())
    autoencoder.add(Dense(nb_classes, activation='sigmoid'))    # für binäre Klassifikation

    return autoencoder
    
def create_encoding_layers(img_size, num_channels):
        kernel = 3
        filter_size = 64
        pad = 1
        pool_size = 2
        return [
            ZeroPadding2D(padding=(pad,pad), input_shape=(num_channels, img_size[0], img_size[1])),
            GaussianNoise(sigma=0.3),   # gegen overfitting
            Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),
    
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(128, kernel, kernel, border_mode='valid'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),
    
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(256, kernel, kernel, border_mode='valid'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),
    
            ZeroPadding2D(padding=(pad,pad)),
            Convolution2D(512, kernel, kernel, border_mode='valid'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),
        ]
    
def create_decoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return[
        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
    ]
