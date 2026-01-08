import os, re, random, shutil, PIL, keras, architekturen
import numpy as np

from math import ceil
from keras.optimizers import SGD

from inspect import getmembers, isfunction

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold
from sklearn.metrics import matthews_corrcoef, hamming_loss, classification_report

from keras.models import model_from_json
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array, array_to_img, load_img, ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
    
def prep_data(data_dict):
    '''Strukturiert die Bilder im Quell-Ordner, falls noch nicht geschehen, um.
       Dazu müssen alle Bilder entweder, nach Klassen getrennt, in Ordnern
       oder einzeln vorliegen. Ein Mix kann nicht verarbeitet werden'''
    
    path = data_dict['path']

    liste = np.asarray(os.listdir(path))                    # Liste mit allen Bildern/Ordnern

    if all(file.is_file() for file in os.scandir(path)):    # Datein befinden sich nicht Ordnern (unsortiert)
        
        abc = defaultdict(list)
        for file in os.scandir(path):                       # Für alle Datein im Ordner
            
            klasse = re.split(r'(\d+)', file.name)[0]       # Klasse aus Bildnamen lesen
            abc[klasse].append(file.name)                   # Nildbezeichnung
            
        max_num = max([len(abc.get(key)) for key in abc.keys()])    # max. Anzahl an Bildern pro Klasse
        
        for klasse in abc.keys():                           # über alle Klassen
            
            for i, file in enumerate(abc[klasse]):          # über alle Dateien der jeweiligen Klasse
            
                subfolder = os.path.join(path,klasse)          # Klassenordner
            
                if not os.path.exists(subfolder):           # Existiert nicht?
                    os.makedirs(subfolder)                  # dann erstelle Ordner
                    
                new_name = klasse + str(i) + '.png'         # Bezeichung des neuen Bildes
                    
                shutil.move(os.path.join(path, file) , os.path.join(path,klasse,new_name)) # Bild (mit neunem Namen) verschieben
                
    elif all(file.is_dir() for file in os.scandir(path)):   # Datein befinden sich in Ordnern (sortiert)
        max_num = 0
        
        for d in liste:                                     # über alle Ordner
            
            for i, file in enumerate(os.listdir(os.path.join(path, d))):    # über alle Dateien der jeweiligen Klasse

                new_name = d +str(i) + '.png'                               # Bezeichung des neuen Bildes

                if not os.path.exists(os.path.join(path, d, new_name)):     # Datei existiert bereits

                    shutil.move(os.path.join(path, d, file) , os.path.join(path, d, new_name))    # Bild (mit neunem Namen) verschieben
                else:
                    continue

            if i+1 >= max_num:
                max_num = i+1
                
    else:                                                   # Mix aus Ordner und Einzeldateien   
        error_msg = 'Inkompatible Dateistruktur'
        
        return error_msg
                
    data_dict['klassen'] = sorted(list(os.listdir(path)))                   # Alle Klassen in eine Liste
    data_dict['noppc'] = max_num

    snoppc(path, max_num)
                
    return data_dict

def snoppc(path, noppc):
    '''same number of pictures per class (snoppc) sorgt dafür, dass für
       jede gleiche die gleiche Anzahl an Bildern zur Verfügung stehen'''
    
    liste = np.asarray(os.listdir(path))

    for d in liste:                         # für jede Klasse
        
        num_of_pics = len(os.listdir(os.path.join(path, d)))    # momentane Bildzahl
        
        if num_of_pics == noppc:            # Anazhl passt 
            continue
        else:                               # Anzahl geringer
            for copy_index, index in enumerate(range(num_of_pics, noppc)):      # über die Differenz
                original = os.path.join(path, d, d + str(copy_index) +'.png')   # Pfad zu Originalbild
                copy_pic = os.path.join(path, d, d + str(index) +'.png')        # Pfad zu Kopie
                shutil.copy(original, copy_pic)                                 # Bild duplizieren      

def transplit(data_dict):
    '''Teilt die Bilder im angegeben Train-Test-Verhältnis (ratio) auf und
       speichert diese, nach Klassen getrennt, in den Ordnern "train" und "test"
       ab'''
    
    path = data_dict['path'] 
    noppc = data_dict['noppc']
    klassen = data_dict['klassen']
    ratio = data_dict['ratio']
    seed = data_dict['seed']

    # Split für Training und Test
    var_A = KFold(n_splits=int(1/ratio), shuffle=True, random_state=seed)
    splitGen = var_A.split(range(noppc))

    # var_B = StratifiedShuffleSplit(n_splits=10, test_size=ratio, random_state=seed)
    # splitGen = var_B.split(range(noppc), Y)

    splitRes = []               
    for split in splitGen:
        splitRes.append(split)      # Liste mit allen Splits

    # nur für Testzwecke - wird später entfernt
    train = splitRes[0][0]          # Indizes für Trainingsbilder
    test = splitRes[0][1]           # Indizes für Testbilder

    # Bilder für Training kopieren
    folder = 'train'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for index in train:             

        for klasse in klassen:      

            subfolder =  os.path.join(folder,klasse)            # Unterordner für jede Klasse

            filename = klasse + str(index) + '.png'             # Pfad zu Originalbild

            if not os.path.exists(subfolder):                   # Unterordner nicht vorhanden
                os.makedirs(subfolder)                          # Unterordner erstellen

            shutil.copy2( os.path.join(path,klasse,filename) , os.path.join(subfolder,filename))    # Originalbild kopieren

    # Bilder für Test kopieren (Erklärung identisch zu Training)
    folder = 'test'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for index in test:

        for klasse in klassen:

            subfolder =  os.path.join(folder,klasse) 

            filename = klasse + str(index) + '.png'

            if not os.path.exists(subfolder):
                os.makedirs(subfolder) 

            shutil.copy2( os.path.join(path,klasse,filename) , os.path.join(subfolder,filename))

    return len(train)*len(klassen), len(test)*len(klassen), splitRes

def test(GUIref, mode):
    '''Klassifiziert ein Bild, welches ausgewählt oder zufällig mit 
       "random Test" erstellt wurde und gibt das Ergebnis, mit entsprechender
       Farbe, aus:
       Grün: Klassifizierung stimmt mit original überein
       Gelb: Klassifizierung hat mind. 1 Fehler zu viel oder zu wenig erkannt
       Rot: Kein klassifizierter Fehler ist im Original vorhanden'''

    data_dict = GUIref.data_dict

    model = data_dict['model']    
    klassen = data_dict['klassen']
    best_threshold = data_dict['best_threshold']
    img_size = data_dict['img_size']
    gray = data_dict['gray'] 
    
    if mode is 'random':                                    # Button "random Test"
        img, k1, k2 = add_rnd_imgs('test', img_size, gray)          
        orig = [k1, k2]                                     # im Bild enthaltene Klassen
    else:                                                   # Button "Bild laden + testen"
        path = GUIref.path2pic                              
        img = load_img(path, grayscale=gray, target_size=img_size)   # Bild laden

        a = os.path.split(path)[1]
        orig = re.split(r'(\d+)', a)[0].split('_')          # im Bild enthaltene Klassen

    plot_img = img.resize((250,360), PIL.Image.ANTIALIAS)  # Bild an Canvasgröße anpassen 
    plot_img = plot_img.convert('L')
    plot_img = plot_img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        
    img = img_to_array(img)             # Bild als Array
    img = img/255                       # Normalisierung
    img = np.expand_dims(img,axis=0)    # Modell erwartet 4 Dimensionen

    pred = model.predict(img)           # Vorhersage

    # Bei allen Klassen, die einen bestimmten Wert (Wahrscheinlichkeit) überschreiten, wird der
    # Wert für die Klasse auf 1 (wahr) gesetzt, ansonsten 0 und anschließend in die Klassen "übersetzt"
    y_pred = np.array([1 if pred[0,i]>=best_threshold[i] else 0 for i in range(pred.shape[1])])
    out = [klassen[i] for i in range(len(klassen)) if y_pred[i]==1 ] 

    set_A = set(orig)           # Im Bild enthaltene Fehler
    set_B = set(out)            # vorhergesagte Fehler

    same = set_A & set_B        # Schnittmenge

    if set_A == set_B:          # Istwert = Sollwert
        color = 'green'
    else:
        if len(same) >= 1:      # 1 Fehler zu viel bzw. zu wenig vorhergesagt
            color = 'yellow'
        else:
            color = 'red'       # mind. 1 Fehler zu viel bzw. zu wenig vorhergesagt


    return plot_img, orig, out, color
    
def predictions_onefold(model, test_generator, data_dict):
    '''Berechnet den optimalen Schwellenwert für jede Klasse und die Genauigkeit
       der Klassifikation anhand der zur Verfügung stehenden Test-Bilder'''

    X_test = []
    Y_test = []

    for i in range(ceil(data_dict['testLen']/data_dict['nb_batch'])):
        data, label = test_generator.next()
        X_test.append(data)
        Y_test.append(label)

    X_test = np.vstack(X_test)                      # Array mit allen Bildern
    Y_test = np.vstack(Y_test)                      # Array mit den echten Labels

    out = model.predict(X_test)                     # Wahrscheinlichkeit für jede Klasse
    out = np.array(out)                             # für jedes Bild in X_test

    threshold = np.arange(0.001,1.0,0.001)          # Suchbereich für Schwellenwert (Wahrscheinlichkeit)

    acc = []                                        # Hilfsvariable
    best_threshold = np.zeros(out.shape[1])         # "idealer" Schwellenwert für jede Klasse
    for i in range(out.shape[1]):                   # für jede Klasse
        y_prob = np.array(out[:,i])                 # Wahrscheinlichkeiten für jedes Bild

        for j in threshold:                         # über den ganzen Suchbereich
            y_pred = [1 if prob>=j else 0 for prob in y_prob]   # Wert auf 1 setzen wenn Wahrscheinlichkeit größer als Schwellenwert, sonst 0
            acc.append(matthews_corrcoef(Y_test[:,i],y_pred))   # Korrelationskoeffizient als Maß für die Güte
        acc   = np.array(acc)                       # Liste zu array konvertieren
        index = np.where(acc==acc.max())            # Index des Schwellenwerts im Suchraum   
        best_threshold[i] = threshold[index[0][0]]  # Schwellenwert aktualisieren
        acc = []

    # Vorhersage der Fehler für jedes Bild, basierend auf den Schwellenwerten der jeweiligen Klasse
    # Array mit den klassifizierten Labels
    y_pred = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j in range(Y_test.shape[1])] for i in range(len(Y_test))])

    print(classification_report(Y_test, y_pred, target_names=data_dict['klassen']))

    # Anazhl der richtig klassifizierten Bilder
    total_correctly_predicted = len([i for i in range(len(Y_test)) if (Y_test[i]==y_pred[i]).sum() == len(data_dict['klassen'])])

    res_acc = model.evaluate(X_test, Y_test)        
    res_acc = "{0:.2f}".format(res_acc[1]*100)      # formatierte Testgenauigkeit 

    data_dict['best_threshold'] = best_threshold
    data_dict['res_acc'] = res_acc

    return data_dict

def predictions_twofold(model, multi_gen, data_dict):
    '''Auswertung für Bilder mit zwei Fehlern. Berechnet werden folgende Werte:
       (Hamming) loss: Bruchteil der Labels die falsch klassifiziert wurden
       2r: Klassifizierung stimmt mit Original überein
       1r: Genau einer von zwei Fehlern wurde richtig erkannt
       0r: Klassifizierung und Original stimmen überhaupt nicht überein
       1r_1f: Mehr als einen Fehler klassifiziert aber nur einer richtig'''
    
    best_threshold = data_dict['best_threshold']
    klassen = data_dict['klassen']

    class_indices = encode(klassen)

    d = defaultdict(float)
    
    for i, item in enumerate(multi_gen):
        data, label = item[0], item[1]
        data = np.expand_dims(data,axis=0)      # Batch-Dimension

        pred = model.predict(data)              # Vorhersage
        y_pred = np.array([1 if pred[0,i]>=best_threshold[i] else 0 for i in range(pred.shape[1])])
        out = [klassen[i] for i in range(len(klassen)) if y_pred[i]==1 ]

        binary_label = np.zeros(len(klassen))
        for klasse in label:
            binary_label += class_indices[klasse]

        d['loss'] += hamming_loss(binary_label, y_pred)

        set_A = set(label)          # Im Bild enthaltene Fehler
        set_B = set(out)            # vorhergesagte Fehler

        same = set_A & set_B        # Schnittmenge

        if set_A == set_B:          # Istwert = Sollwert
            d['2r'] += 1
        else:
            if len(same) == 1:      
                if len(set_B) == 1: # genau 1 fehler erkannt
                    d['1r'] += 1
                else:               # 1 Fehler erkannt - 1 falsch vorhergesagt
                    d['1r_1f'] += 1             
            else:
                d['0r'] += 1       # komplett falsch

    res_string = []
    d['loss'] /= i+1

    for k, v in sorted(d.items(), reverse=True):
        if k is 'loss':
            res_string.append('{} : {}\n\n'.format(k, round(v, 4)))
        else:
            res_string.append('{} : {} ({} %)\n'.format(k, v, round(v/(i+1)*100,1)))
            
    data_dict['d'] = d
    data_dict['res_string'] = res_string
                
    return data_dict

def load_model(path2model):
    '''Lädt das Modell mit Hilfe von json'''
    
    json_file = open(path2model, 'r')                   # Modell mit json öffnen
    loaded_model_json = json_file.read()                # Modell auslesen
    json_file.close()                                   # json schließen
    loaded_model = model_from_json(loaded_model_json)   # Modell laden
    
    return loaded_model

def make_generators(data_dict):
    '''Prodiziert Generatoren für die Training- und Testbilder'''
    
    img_size = data_dict['img_size']  
    nb_batch = data_dict['nb_batch']

    gray = data_dict['gray'] 

    if gray == True:
        mode = 'grayscale'
    else:
        mode = 'rgb'

    # Train Augmentaion
    train_datagen = ImageDataGenerator(rescale=1./255,
                    rotation_range=15,
                    width_shift_range=0.1,
                    height_shift_range=0.1)

    # Test Augmentaion
    test_datagen = ImageDataGenerator(rescale=1./255)

    # produziert Bilder für Training in Echtzeit
    train_generator = train_datagen.flow_from_directory(
        'train',  
        color_mode=mode,
        target_size=img_size,  
        batch_size=nb_batch,
        class_mode='categorical')

    # produziert Bilder für Test in Echtzeit
    test_generator = test_datagen.flow_from_directory(
        'test',
        color_mode=mode,
        target_size=img_size,
        batch_size=nb_batch,
        class_mode='categorical')

    return train_generator, test_generator

def multi_class_image_gen(data_dict, src_folder):
    '''Alle Bilder im Quell-Ordner (in diesem Fall "test") werden miteinander
       kombiniert (nicht redundant) und in einem Generator gespeichert, um sie
       später auszuwerten'''
    
    klassen = data_dict['klassen'] 
    img_size = data_dict['img_size'] 
    gray = data_dict['gray'] 

    liste = []                      # Liste mit allen Bildern

    for path, subdirs, files in os.walk(src_folder):            # Für alle Bilder
        for name in files:
            
            klasse = path.split('\\')[-1]                       # Klasse bzw. Unterordner
            
            liste.append([klasse, os.path.join(path, name)])    # Klasse + Bildpfad
            
        durch = []                  # Liste bereits verwendeter Klassen
            
    for klasse in klassen:
        
        combNum = 0                 # Überschreibungen vermeiden
        
        durch.append(klasse)    
        
        combo_A = [ num for num in liste if num[0] == klasse ]      # Bilder einer Klasse
    
        combo_B = [ num for num in liste if num[0] not in durch ]   # Bilder der noch nicht verwendeten Klassen
    
        if combo_A == combo_B:      # nicht mit sich selbst kombinieren
            break
    
        for file_A in combo_A:      
            
            for file_B in combo_B:
                
                    img1 = img_to_array(load_img(file_A[1], grayscale=gray, target_size=img_size))  # 1. Bild zu Array konvertieren (für Addition)
                    img2 = img_to_array(load_img(file_B[1], grayscale=gray, target_size=img_size))  # 2. Bild zu Array konvertieren (für Addition)

                    img = (img1 + img2) / 2                     # Bilder überlagern
                    img = img/255                               # normalisieren
                
                    dis_img = array_to_img(img)                 # Bild zurück konvertieren (für Darstellung)
                    
                    dis_img.save('combo/' + file_A[0] + '_' + file_B[0] + str(combNum) + '.png')

                    label = [file_A[0], file_B[0]]

                    combNum += 1
                    
                    yield img, label                            # Namen der Klassen + Bild

def encode(klassen):
    '''Jeder Klasse (key), wird in einem dictionary, die entsprechende 
       1-aus-n-codierte Repräsentation (value) zugewiesen'''
    
    encoder = LabelEncoder()
    encoder.fit(klassen)
    encoded = encoder.transform(klassen)            # Klassennamen als Integer codieren

    dummy_y = np_utils.to_categorical(encoded)      # Indizematrix für die Integer

    class_indices = dict(zip(klassen, dummy_y))     # dictionary für leichtere Zuweisung

    return class_indices

def get_models(GUIref):
    '''Liest die Modelle (Funktionen) aus dem entsprechenden Script und schreibt 
       diese in das Drop-Down-Menü "Modell" im Hauptfensters'''

    nicht = ['__init__', 'create_model']        # Hilfsfuktionen nicht beachten

    # Liste mit allen zur Verfügung stehenden Modellen
    functions_list = [o[0] for o in getmembers(architekturen.Model_Creator, isfunction) if o[0] not in nicht]

    GUIref.drop_model.addItems(functions_list)  # Liste an Drop-Down-Menü übergeben

def add_rnd_imgs(path, img_size, gray):
    '''Vereint zwei zufällig ausgewählte Bilder aus zwei zufällig
       ausgewählten Fehlern in einem Bild'''

    dirs = os.listdir(path)                                             # Liste mit allen Klassen

    klasse_1 = random.choice(dirs)                                      # 1. Klasse zufällig auswählen
    bild_1 = random.choice(os.listdir(os.path.join(path, klasse_1)))    # 1. Bild zufällig auswählen

    dirs.remove(klasse_1)                                               # Vermeiden, dass 2 mal die gleiche Klasse verwendet wird

    klasse_2 = random.choice(dirs)                                      # 2. Klasse zufällig auswählen
    bild_2 = random.choice(os.listdir(os.path.join(path, klasse_2)))    # 2. Bild zufällig auswählen

    img_1 = img_to_array(load_img(os.path.join(path, klasse_1, bild_1), grayscale=gray, target_size=img_size))      # 1. Bild zu Array konvertieren (für Addition)
    img_2 = img_to_array(load_img(os.path.join(path, klasse_2, bild_2), grayscale=gray, target_size=img_size))      # 2. Bild zu Array konvertieren (für Addition)

    img = (img_1 + img_2) / 2           # Bilder überlagern

    img = array_to_img(img)             # Bild zurück konvertieren (für Darstellung)
    
    return img, klasse_1, klasse_2

def start_new():
    '''Löscht die temporären Ordner und stellt sie anschließend wieder her,
       um zu gewährleisten, dass immer die richtigen Daten vorliegen'''

    # temporäre Ordner, falls vorhanden, löschen
    if os.path.isdir('train'): 
        shutil.rmtree('train')  
    if os.path.isdir('test'):   
        shutil.rmtree('test')   
    if os.path.isdir('combo'):  
        shutil.rmtree('combo') 

    # temporäre Ordner erstellen
    os.makedirs('train')
    os.makedirs('test')
    os.makedirs('combo')

def test_folder(GUIref, src_folder):
    '''Testet den ausgewählten Ordner und gibt die Genauigkeit des
       Modells für diesen an'''

    data_dict = GUIref.data_dict

    model = data_dict['model']    
    klassen = data_dict['klassen']
    img_size = data_dict['img_size']
    gray = data_dict['gray']

    lrate = data_dict['lrate'] 
    decay = data_dict['decay']
    nestorov = data_dict['nestorov']
    momentum = data_dict['momentum']

    prep_data(data_dict)

    encoder = encode(klassen)

    data, labels = [], []

    for path, subdirs, files in os.walk(src_folder):    # Für alle Bilder
        for name in files:
            
            klasse = path.split('\\')[-1]               # Klasse bzw. Unterordner
            
            labels.append(encoder[klasse])              # Klasse + Bildpfad
            
            img = img_to_array(load_img(os.path.join(path, name), grayscale=gray, target_size=img_size))

            img = img/255                               # normalisieren (wichtig!)

            img = np.expand_dims(img,axis=0)            # Modell erwartet 4 Dimensionen

            data.append(img)
            
    X_test = np.vstack(data)                            # Array mit allen Bildern
    Y_test = np.vstack(labels)                          # Array mit den echten Labels

    sgd = SGD(lr=lrate, momentum=momentum, decay=decay, nesterov=nestorov)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['fbeta_score'])

    res_acc = model.evaluate(X_test, Y_test)
    res_acc = "{0:.2f}".format(res_acc[1]*100)          # formatierte Testgenauigkeit 

    return res_acc

def get_y_ticks_10():
	"""Y-Achsenbeschriftung für Anzeige. Stellen für 10er Potenzen finden.
	"""
	#Das ist noch verschoben weil keine richtige 0 da ist

	log_factor = 54.0691				#nur für log wichtig


	y_labels = np.logspace(0, 6, num=7, dtype=np.int)
	y_labels = np.round(y_labels)

	y_labels = [i for i in y_labels if i <= 10000]


	y_ticks = []
	for item in y_labels:
		y_ticks.append(np.round(np.log10(item)*log_factor))


	return y_ticks, y_labels

class LossHistory(keras.callbacks.Callback): 
    '''Callback-Funktion, die den Ladebalken im Hauptfenster updatet'''
    
    def __init__(self, progressbar=None):   
        self.progressbar = progressbar          # Referenz auf Ladebalken

    def on_train_begin(self, logs={}):                                        
        self.losses = []                        # Liste bzw. Länge als Referenzwert 

    def on_epoch_end(self, progressbar, logs={}):
        self.losses.append(logs.get('loss'))    # Länge entspricht der momentanen Epoche

        nb_epoch = self.params['nb_epoch']      # Anzahl Epochen

        print('Lernrate: {}'.format(keras.backend.get_value(self.model.optimizer.lr)))

        prozent = len(self.losses)/nb_epoch*100 # Status in Prozent 

        self.progressbar.setProperty("value", prozent)  # update Ladebalken

