import os
    
os.environ['KERAS_BACKEND'] = 'tensorflow'

import helfer as h
import architekturen as a

import keras

def train(GUIref):

    data_dict = GUIref.data_dict
    progressbar = GUIref.progressBar
    status = GUIref.status

    status.setText("Simulation gestartet")   
    
    nb_epoch = data_dict['nb_epoch']        # Anzahl der Durchläufe
    model_type = data_dict['model_type']    # ausgewähltes Modell

    h.start_new()

    data_dict = h.prep_data(data_dict)
    if isinstance(data_dict, str): return data_dict

    # Verschiedene Ordner für Test und Training + Klassen
    if status: status.setText("Daten aufbereiten")
    data_dict['trainLen'], data_dict['testLen'], data_dict['splitRes'] = h.transplit(data_dict)

    # Generatoren für Training- und Testbilder erzeugen
    if status: status.setText("Erzeuge Generatoren")
    train_generator, test_generator = h.make_generators(data_dict)
    multi_generator = h.multi_class_image_gen(data_dict, src_folder='test')

    # Modell erstellen
    if status: status.setText("Initilaisiere Modell")
    creator = a.Model_Creator(data_dict, train_generator)
    model = creator.create_model(model_type)
    
    # learning-rate vermindern bei Stagnation
    reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                  patience=10, min_lr=0.00001)

    # Frühzeitig abbrechen, wenn keine Verbesserung
    #early_stop = keras.callbacks.EarlyStopping(min_delta=0.001, patience=20, verbose=0, mode='auto')

    callbacks_list = [reduce_lr]

    if progressbar: callbacks_list.append(h.LossHistory(progressbar))
    
    # Modell trainieren
    if status: status.setText("Trainiere Modell")
    history = model.fit_generator(
        train_generator,
        validation_data=test_generator,
        nb_val_samples=data_dict['testLen'],
        verbose=2,
        samples_per_epoch=data_dict['trainLen'],
        callbacks=callbacks_list,
        nb_epoch=nb_epoch)

    # Verlauf der Genauigkeit plotten
    # fig = plt.figure()
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    # Gewichte
    weights = model.get_weights()
    
    # Vorhersagen über Testbilder etc. berechnen
    data_dict = h.predictions_onefold(model, test_generator, data_dict)
    data_dict = h.predictions_twofold(model, multi_generator, data_dict)

    data_dict['model'] = model
    data_dict['weights'] = weights

    return data_dict