import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt, numpy as np
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys, gui, TE_CNN, helfer, pickle, re, time, keras

class MainUiClass(QtWidgets.QMainWindow, gui.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainUiClass, self).__init__(parent)
        self.setupUi(self)

        helfer.get_models(self)                             # Liste mit Modellen

        self.progressBar.setProperty("value", 0)  			# Ldebalken auf 0 setzen
        self.p_picPath.clicked.connect(self.get_files)		# Button "Bildverzecihnis"
        self.p_train.clicked.connect(self.train)			# Button "Training"
        self.p_saveW.clicked.connect(self.saveW)			# Button "Gewichte speichern"
        self.p_loadW.clicked.connect(self.loadW)			# Button "Gewichte laden"
        self.p_saveM.clicked.connect(self.saveM)			# Button "Modell speichern"
        self.p_loadM.clicked.connect(self.loadM)			# Button "Modell laden"
        self.p_rand_test.clicked.connect(self.test_random)  # Button "random Test"
        self.p_select_test.clicked.connect(self.test_pic)	# Button "Bild laden + testen"
        self.p_tes_folder.clicked.connect(self.test_folder)

        self.data_dict = {}                                 # Speicher-Variable initilisieren

        self.actionthread = CustomThread(self.train)        # Training in neuem Thread (update des Ladebalkens)

        # Größe des Hauptfensters anpassen (rein ästhetisch)
        X = self.label.x()
        weite = self.label.width()
        Y = self.label.y()
        hoehe = self.label.height()

        self.resize(X+weite+5, Y+hoehe+20)

        self.canvas = PlotCanvas(self, width=weite/100, height=hoehe/100)       # Figure mit weite*hoehe Pixeln
        self.canvas.move(X, Y)                                          # an der Stelle X, Y 
        self.canvas.setup()                                             # Canvas initialisieren

    def closeEvent(self, event):
        '''Schließt die GUI'''
        event.accept()

    def get_files(self):
        '''Speichert den Pfad zum ausgewählten Quell-Verzeichnis ab'''

        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        folder = QFileDialog.getExistingDirectory(dlg, 'Ordner auswählen')
        self.data_dict['path'] = folder + "/"
        self.p_train.setEnabled(True)

    def get_data(self):
        '''Fasst alle Parameter in einem dictionary zusammen'''

        self.data_dict['model_type'] = str(self.drop_model.currentText())

        self.data_dict['gray'] = self.radio_gray.isChecked()
        self.data_dict['nestorov'] = self.radio_nestorov.isChecked()
        self.data_dict['visualize'] = self.radio_plot.isChecked()

        self.data_dict['ratio'] = float(self.edit_ratio.text())
        self.data_dict['nb_epoch'] = int(self.edit_epochs.text())  
        self.data_dict['img_size'] = tuple([int(s) for s in re.findall(r"\d+", self.edit_size.text())]) 
        self.data_dict['nb_batch'] = int(self.edit_batch.text())

        self.data_dict['lrate'] = float(self.edit_lrate.text())
        self.data_dict['decay'] = float(self.edit_decay.text())
        self.data_dict['momentum'] = float(self.edit_momentum.text())

        self.data_dict['seed'] = 8008

    def train(self):
        '''Modell mit ausgewählten Einstellungen trainieren'''

        start = time.clock()   

        self.get_data()

        try:
            data_dict = TE_CNN.train(self)
        except KeyError:
            self.popup_error('Es wurde noch kein Bildverzeichnis ausgewählt')
            self.status.setText("Kein Bildverzeichnis")
            return
        except MemoryError:
            self.popup_error('Nicht genug Speicher\nBatch- oder Bild-Größe verringern')
            return

        if isinstance(data_dict, str): 
            self.popup_error(data_dict) 
            return
        else:
		
            self.data_dict = data_dict   
            self.model = data_dict['model']     

            self.label_time.setText('Zeit: {} s'.format(round(time.clock()-start, 2)))
            self.label_acc.setText('Genauigkeit (1 F): ' +  data_dict['res_acc'] + ' %')
            self.label_acc_2.setText('Genauigkeit (2 F):\n' + "".join(data_dict['res_string']))

            self.enable_buttons()

            self.status.setText("Simulation abgeschlossen") 

    def test_pic(self):
        '''Gibt das Ergebnis der Klassifikation an und plottet das ausgewählte Bild'''

        dlg = QFileDialog()
        path2pic = QFileDialog.getOpenFileName(dlg, "Bild wählen")
        self.path2pic = path2pic[0]
        
        img, orig, pred, color = helfer.test(self, 'chosen')

        # Ergebnisse mit entsprechenden Farben (je nach Übereinstimmung) anzeigen
        self.label_orig.setText('Original: \n' + " ".join(orig))
        self.label_orig.setStyleSheet('background-color: ' + color)
        self.label_pred.setText('Vorhersage: \n' + " ".join(pred))
        self.label_pred.setStyleSheet('background-color: ' + color)

        self.canvas.plot(img)   # Bild plotten

    def test_random(self):
        '''Gibt das Ergebnis der Klassifikation an und plottet das zufällig erstellte Bild'''

        img, orig, pred, color = helfer.test(self, 'random')

        # Ergebnisse mit entsprechenden Farben (je nach Übereinstimmung) anzeigen
        self.label_orig.setText('Original: \n' + " ".join(orig))
        self.label_orig.setStyleSheet('background-color: ' + color)
        self.label_pred.setText('Vorhersage: \n' + " ".join(pred))
        self.label_pred.setStyleSheet('background-color: ' + color)

        self.canvas.plot(img)	# Bild plotten

    def test_folder(self):
        '''Testet den ausgewählten Ordner und gibt die Genauigkeit des
           Modells für diesen an'''

        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        folder = QFileDialog.getExistingDirectory(dlg, 'Ordner auswählen')

        res_acc = helfer.test_folder(self, folder)

        self.label_acc.setText('Genauigkeit (1 F): ' +  res_acc + ' %')


    def saveW(self):
        '''Gewichte + Einstellungen abspeichern'''

        dlg = QFileDialog()
        path2w = QFileDialog.getSaveFileName(dlg, "Speicherort wählen")[0]

        if path2w.find('.p') == -1:
            path2w += '.p'
  
        self.data_dict.pop('model', None)                   # Modell nicht mit abspeichern

        pickle.dump(self.data_dict, open(path2w, "wb"))	# Dict. abspeichern

    def loadW(self):
        '''Gewichte + Einstellungen laden'''

        dlg = QFileDialog()
        path2w = QFileDialog.getOpenFileName(dlg, "Gewichte wählen", filter="*.p")[0]

        if not path2w:
            self.status.setText("Keine Gewichte geladen")
        else:
        
            with open(path2w, 'rb') as fp:			# dict. laden
                self.data_dict = pickle.load(fp)

            self.model.set_weights(self.data_dict['weights'])

            self.data_dict['model'] = self.model
    			
            self.enable_buttons()
        
    def saveM(self):
        '''Modell abspeichern'''

        dlg = QFileDialog()
        path2m = QFileDialog.getSaveFileName(dlg, "Speicherort wählen")[0]

        if path2m.find('.json') == -1:
            path2m += '.json'

        with open(path2m, "w") as json_file:	# Modell speichern
            json_file.write(self.model.to_json())     

    def loadM(self):
        '''Modell laden'''

        dlg = QFileDialog()
        path2m = QFileDialog.getOpenFileName(dlg, "Modell wählen", filter="*.json")[0]

        if not path2m:
            self.status.setText("Kein Modell geladen")
        else:

            self.model = helfer.load_model(path2m)	# Modell laden
            
            self.data_dict['model'] = self.model

            self.p_loadW.setEnabled(True)

    def popup_error(self, msg):

        QMessageBox.critical(self, 'Fehler!', msg, QMessageBox.Ok)

    def enable_buttons(self):
        self.p_loadW.setEnabled(True)               # Button "Gewichte laden"
        self.p_saveW.setEnabled(True)               # Button "Gewichte speichern"
        self.p_saveM.setEnabled(True)               # Button "Modell speichern"
        self.p_rand_test.setEnabled(True)           # Button "random Test"
        self.p_select_test.setEnabled(True)         # Button "Bild laden + testen"
        self.p_tes_folder.setEnabled(True)

class CustomThread(QtCore.QThread):
    def __init__(self, target, slotOnFinished=None):
        super(CustomThread, self).__init__()
        self.target = target
        if slotOnFinished:
            self.finished.connect(slotOnFinished)

    def run(self, *args, **kwargs):
        self.target(*args, **kwargs)

class PlotCanvas(FigureCanvas):
 
    def __init__(self, parent=None, width=5, height=4, dpi=100, frameon=False):	# Standardwerte
        fig = plt.figure(figsize=(width, height), dpi=dpi, frameon=frameon)		# Übergebene Werte
 
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,	# Verhalten bei Resizing (dynamisch)
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.figure = fig 			
    
    def setup(self):

        image = np.zeros((250,360))
        image[0,0] = 255

        ax = self.figure.add_subplot(111)               # Achse für plot

        # Achsenbeschriftungen und -Ticks (X-Achse)

        ax.set_xlabel('Phasenwinkel ' + r'$ \varphi$')
        ax.set_xlim(xmin=0, xmax=360)

        x_ticks = np.arange(0, 361, 90)
        x_labels = [str(s) for s in x_ticks]

        ax.set_xticks(x_ticks) 
        ax.set_xticklabels(x_labels)

        y_ticks, y_labels = helfer.get_y_ticks_10()

        # Achsenbeschriftungen und -Ticks (Y-Achse)
        ax.set_ylabel('Entladungsstärke ' + r'$p/pC$')
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

        mycmap = cm.get_cmap('viridis')                 # verwendete Colormap
        mycmap.set_under('w')                           # alle Bereiche unter Schwellenwert (1) weiß anzeigen

        im = ax.imshow(image, cmap=mycmap, vmin=1, interpolation='nearest')

        # Colorbar hinzufügen
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = self.figure.colorbar(im, cax=cax, ticks=[1, 128, 255], orientation='vertical')
        cbar.ax.set_yticklabels(['wenig', 'mittel', 'viel'])
        cbar.ax.set_ylabel('Anzahl an Teilentladungen')
        self.ax = ax

        ax.invert_yaxis()
        ax.set_ylim(ymin=0)

        self.draw()                                     # an Canvas übergeben

        plt.tight_layout()  

    def plot(self, image):
    	# Getestetes Bild anzeigen

        mycmap = cm.get_cmap('viridis')                 # verwendete Colormap
        mycmap.set_under('w')                           # alle Bereiche unter Schwellenwert (1) weiß anzeigen

        im = self.ax.imshow(image, cmap=mycmap, vmin=1, interpolation='nearest', aspect='auto')
        plt.savefig('hmm.png')

        self.draw()			                            # an Canvas übergeben
 
if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    myapp = MainUiClass()
    myapp.show()
    app.exec_()



    
