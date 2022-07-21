import os
import gc
import mne
import glob
import warnings
import numpy as np
import scipy.io as sio
from typing import List, Optional, Tuple


class SplitData:
    """Clase para generar archivos npz con formato para ML
    """
    
    def __init__(self, subject:str,
                 type_action:Optional[str]='thinking_inds',
                 ignore_channels:Optional[List[str]]=['VEO', 'HEO','EKG', 'EMG','Trigger'],
                 keep_channels:Optional[List[str]]=[],
                 root_folder:Optional[str]='Data',
                 src_folder:Optional[str]='RawDataExtracted',
                 dst_folder:Optional[str]='SplittedData',
                 ):
        """Constructor de la clase

        Args:
            subject (str): Sujeto a procesar con el nombre de KaraOne
            type_action (Optional[str], optional): Accion que buscamos de los sujetos de KaraOne. Defaults to 'thinking_inds'.
            ignore_channels (Optional[List[str]], optional): Canales a ignorar del eeg. Defaults to ['VEO', 'HEO','EKG', 'EMG','Trigger'].
            keep_channels (Optional[List[str]], optional): Canales que queramos mantener. Defaults to [].
            root_folder (Optional[str], optional): Folder donde estan las carpetas. Defaults to 'Data'.
            src_folder (Optional[str], optional): Folder donde estan los datos de KaraOne. Defaults to 'RawDataExtracted'.
            dst_folder (Optional[str], optional): Folder donde guardaremos el NPZ. Defaults to 'SplittedData'.

        Raises:
            ValueError: Retorna si type_action no es valido. ['thinking_inds', 'clearing_inds']
        """
        
        self.subject = subject
        
        self.subject_path = os.path.join(root_folder, src_folder, subject)
        
        self.output_path = os.path.join(root_folder, dst_folder)
        
        self.output_file = os.path.join(self.output_path, f"{subject}.npz")
        
        if type_action not in ['clearing_inds', 'thinking_inds']:
            raise ValueError(f"type_action {type_action} not supported on this dataset")

        else:self.type_target = type_action
        
        self.ignore_channels = ignore_channels
        
        self.keep_channels = keep_channels
        
        self._load_epochs_()
        self._load_targets_()
        self._load_raw_()
        
        self._create_output_folder_()
    
    def _create_output_folder_(self):
        """Se encarga de crear el directorio de salida

        Raises:
            PermissionError: Si no se tiene permisos de escritura en el directorio de salida
        """
        if not os.path.exists(self.output_path):
            
            try:
                os.mkdir(self.output_path)
            
            except PermissionError:
                raise PermissionError(f"Permission denied to create {self.output_path}")
        
        else:
            print(f"{self.output_path} already exists, some files may be overwritten")
    
    def _load_epochs_(self):
        """Se encarga de cargar los indices de las epocas

        Raises:
            FileNotFoundError: Si no se encuentra el archivo de epochs_inds.mat
        """
        
        try:
            self.epochs_index = sio.loadmat(os.path.join(self.subject_path, "epoch_inds.mat"),
                                            variable_names=('clearing_inds',
                                                            'thinking_inds'))
            
        except FileNotFoundError:
            raise FileNotFoundError(f"{self.subject_path}/epoch_inds.mat not found")
    
    def _load_targets_(self):
        """Carga las palabras de los epochs

        Raises:
            FileNotFoundError: Si no se encuentra el archivo de all_features_simple.mat
        """
        try:
            targets = sio.loadmat(os.path.join(self.subject_path, "all_features_simple.mat"))
        
        except FileNotFoundError:
            raise FileNotFoundError(f"{self.subject_path}/all_features_simple.mat not found")
        
        else:
            self.targets = targets['all_features'][0, 0]["prompts"][0]
    
    def _load_raw_(self):
        """Carga el archivo raw de la data
        
        Raises:
            IndexError: Si no se encuentra el archivo del eeg.cnt
        """
        
        print(f"Loading raw data for subject {self.subject}")
        
        try:
            cnt_file = glob.glob(os.path.join(self.subject_path, "*.cnt"))[0]
            
            self.raw = mne.io.read_raw_cnt(cnt_file,
                                           preload = True,
                                           verbose = False,
                                           )
        except IndexError:
            raise IndexError(f"{self.subject_path}/*.cnt not found")
        
        if self.ignore_channels:
            self.raw = self.raw.drop_channels(self.ignore_channels)
        
        if len(self.keep_channels) >= 1:
            self.raw = self.raw.pick_channels(self.keep_channels)
        
    def split(self, duration_time:Optional[int]=4500, save:Optional[bool]=False) -> Tuple:
        """Se encragra de dividir la data en dataset para algoritmos de ML

        Args:
            duration_time (Optional[int], optional): Tiempo de duracion de la accion. Defaults to 4500.
            save (Optional[bool], optional): Guardar los datos crudos en un archivo npz. Defaults to False.

        Raises:
            ValueError: Si el tiempo de duracion es menor a cero o mayor a 5000

        Returns:
            np.ndarray: 3D array de EEG 
            np.ndarray: 1D array de labels
        """
        
        print('Making dataset')
        
        EEG = np.zeros(shape=(len(self.targets), len(self.raw.ch_names),  duration_time))
        
        offset = int(self.raw.info["sfreq"]/2)
    
        for word_index, _ in enumerate(self.targets):
            
            """
            epochs_index[type].shape = 1, 165 1 columna de 165 epocas
            epochs_index[type][word_index][0].shape = 165 que son los trail totales del sujeto
            self.epochs_index[self.type_target][0][word_index].shape = (1,2) Me supongo hace referencia a inicio y fin del pensamiento
            end_epoch = self.epochs_index[self.type_target][0][word_index][0][1] Por el momento no utilizare esta ya que no me da 5 segundos exactos
            """
            
            start_epoch = self.epochs_index[self.type_target][0][word_index][0][0] + offset
            
            if duration_time == 5000:
                end_epoch = self.epochs_index[self.type_target][0][word_index][0][1] + offset
            
            elif duration_time < 5000 and duration_time > 0:
                end_epoch = start_epoch + duration_time
            
            else:
                
                raise ValueError(f"duration_time must be less than 5000")
            
            for channel_index, _ in enumerate(self.raw.ch_names):
                """
                    len(self.raw[channel_index]) = 2 Data, time
                    self.raw[channel_index][0].shape = 1,2477400 Un trail de 2477400 muestras
                    self.raw[channel_index][1].shape = 2477400 Tiempo en ms de toda la muestra  
                    self.raw[channel_index][0][0] = todas las muestras de el trail
                    self.raw[channel_index][0][0][start_epoch:end_epoch].shape = 5000 muestras por epoca
                """
                EEG[word_index, channel_index, 0:end_epoch - start_epoch] = self.raw[channel_index][0][0][start_epoch:end_epoch]
                
        if save:
            np.savez(self.output_file, EEG=EEG, targets=self.targets)
            
            print(f"Dataset saved to {self.output_file}")
            print(f"with shape {EEG.shape}, and {len(self.targets)} targets")
        
            del EEG
            del self.raw
            gc.collect()
        
        else:
            return EEG, self.targets


if __name__ == '__main__':
    
    test = SplitData(subject='MM05',
                     type_action='thinking_inds',
                     ignore_channels=['VEO', 'HEO','EKG', 'EMG','Trigger'],
                     keep_channels=['T7', 'C5', 'C3', 'CP5', 'CP3', 'CP1', 'P3', 'C4', 'FC6', 'FT8'],
                     root_folder='Data',
                     src_folder='RawDataExtracted',
                     dst_folder='SplittedData')
    
    test.split(duration_time=4400, save=True)