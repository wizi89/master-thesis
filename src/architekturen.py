import os, h5py, sys
import helfer as h
import numpy as np
import segnet_model as sm
from ResNet import ResnetBuilder as rb

from keras.models import Sequential
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.layers import Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import img_to_array, array_to_img, load_img, ImageDataGenerator
from keras.applications import inception_v3, vgg16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import matthews_corrcoef

class Model_Creator():
    def __init__(self, data_dict, train_generator):
        self.train_generator = train_generator
        self.data_dict = data_dict

        self.img_size = self.data_dict['img_size']
        self.klassen = self.data_dict['klassen']

        if self.data_dict['gray'] is False: 
            self.num_channels = 3
        else: 
            self.num_channels = 1
        
    def simple(self):
        '''einfache Architektur'''
    
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape=(self.num_channels, self.img_size[0], self.img_size[1]), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.klassen), activation='sigmoid'))        # binäre Klassifikation
    
        return model

    def autoencoder(self):

        model = sm.autoencoder(self.img_size, self.klassen, self.num_channels)

        return model

    def PRPDNet(self):

        model = Sequential()
        model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same',input_shape=(self.num_channels, self.img_size[0], self.img_size[1])))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))

        model.add(Flatten())
        model.add(Dense(len(self.klassen), activation='sigmoid'))

        return model

    def ResNet(self):

        return rb.build((self.num_channels, self.img_size[0], self.img_size[1]), 
                                                       len(self.klassen), 'bottleneck', [3, 4, 23, 3])

        
    def VGG16(self):
        '''VGG16 Architektur'''

        model = Sequential()
        model.add(ZeroPadding2D((1,1), input_shape=(self.num_channels, self.img_size[0], self.img_size[1])))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Flatten())
    
        model.add(Dense(len(self.klassen), activation='sigmoid'))
    
        return model
        
    def pre_VGG16(self):
        '''VGG16 Architektur vortrainiert mit ImageNet-Bildern'''
        
        # Vortrainiertes Modell (mit ImageNet-Gewichten) laden
        base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(self.num_channels, self.img_size[0], self.img_size[1]))
    
        # Modell (bisher rein CNN) um fully-connected (FC) Netzwerk ergänzen (für Klassifikation)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(len(self.klassen), activation='sigmoid')(x)
    
        # Dieses Modell wird am Ende trainiert
        model = Model(input=base_model.input, output=predictions)
    
        # Alle Schichten im vortrainierten Modell statisch lassen
        for layer in base_model.layers:
            layer.trainable = False
    
        # Modell kompilieren
        model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    
        # Modell für einige Epochen trainieren
        model.fit_generator(
            self.train_generator,
            verbose=2,
            samples_per_epoch=100,
            nb_epoch=100)
    
        # Den letzten Block des CNN dynamisch machen
        for layer in model.layers[:14]:
           layer.trainable = False
        for layer in model.layers[14:]:
           layer.trainable = True
    
        # Modell wieder kompilieren (mit sehr geringer learning-rate),
        # damit die Änderungen übernommen werden können
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy')
        
        # Fine-Tuning des letzten CNN-Blocks + FC-Netzwerk
        model.fit_generator(
            self.train_generator,
            verbose=2,
            samples_per_epoch=100,
            nb_epoch=100)
    
        return model
        
    def pre_InceptionV3(self):
        '''InceptionV3 Architektur vortrainiert mit ImageNet-Bildern'''
        
        # vortrainiertes Modell (mit ImageNet-Gewichten) laden
        base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(self.num_channels, self.img_size[0], self.img_size[1]))
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(len(self.klassen), activation='sigmoid')(x)
        
        # Dieses Modell wird am Ende trainiert
        model = Model(input=base_model.input, output=predictions)
        
        # Alle Schichten im vortrainierten Modell statisch lassen
        for layer in base_model.layers:
            layer.trainable = False
        
        # Modell kompilieren
        model.compile(optimizer='rmsprop', loss='binary_crossentropy')
        
        # Modell für einige Epochen trainieren
        model.fit_generator(
            self.train_generator,
            verbose=2,
            samples_per_epoch=100,
            nb_epoch=100)
        
        # Nun sind die oberen Schichten (gut) trainiert und das Fine-Tuning der
        # Convolutional Layers kann beginnen. Dazu werden die obersten zwei
        # Blöcke wieder dynamisch gemacht (Zahlen aus Blick auf die Architektur)
        for layer in model.layers[:172]:
           layer.trainable = False
        for layer in model.layers[172:]:
           layer.trainable = True
        
        # Modell wieder kompilieren (mit sehr geringer learning-rate),
        # damit die Änderungen übernommen werden können
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy')
        
        # Fine-Tuning der letzten beiden CNN-Blocks + FC-Netzwerk
        model.fit_generator(
            self.train_generator,
            verbose=2,
            samples_per_epoch=100,
            nb_epoch=100)
    
        return model
        
    def create_model(self, model_type, visualize=False):
        '''Gibt das zu trainieredne Modell aus, basierend auf den Einstellungen'''
        
        fn = getattr(self, model_type)      # Funktion anhand des ausgewähltem
        model = fn()                        # Modells aufrufen
        
        if self.data_dict['visualize']:
            from keras.utils.visualize_util import plot

            current_dir = os.path.dirname(os.path.realpath(__file__))   # Arbeitsordner
            model_path = os.path.join(current_dir, "architektur.png")   # Pfad zu Modell
            plot(model, to_file=model_path, show_shapes=True)           # Modell plotten und speichern

        lrate = self.data_dict['lrate'] 
        decay = self.data_dict['decay']
        nestorov = self.data_dict['nestorov']
        momentum = self.data_dict['momentum']

        # Modell kompilieren
        sgd = SGD(lr=lrate, momentum=momentum, decay=decay, nesterov=nestorov)
        model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['fbeta_score'])   
    
        return model
    