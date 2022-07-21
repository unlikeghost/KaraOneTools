import os
import gc
import math
import dtcwt
import warnings
import numpy as np
from typing import Optional, Tuple
from pyts.preprocessing import RobustScaler
from pyts.multivariate.transformation import MultivariateTransformer


class Wavelet:
    """Clase encargada de transformar datos crudos de EEG a una DTCWT de 1dim
    """
    
    def __init__(self, subject:str,
                 desc_level:int,
                 root_folder:Optional[str]='Data',
                 src_folder:Optional[str]='SplittedData',
                 dst_folder:Optional[str]='WaveldData'):
        """Constructor de la clase Wavelet

        Args:
            subject (str): Sujeto a procesar con nombrado de forma de KaraOne
            desc_level (int): Nivel de descomposición de la transformada de wavelet
            root_folder (Optional[str], optional): Folder donde estan los otros folders. Defaults to 'Data'.
            src_folder (Optional[str], optional): Folder donde se encuentran los archivos npz. Defaults to 'SplittedData'.
            dst_folder (Optional[str], optional): Folder donde guardaremos el nuevo npz. Defaults to 'WaveldData'.

        Raises:
            FileNotFoundError: Si no se encuentra el archivo npz
        """
        self.working_folder = os.path.join(root_folder, src_folder)
        
        self.data_file = os.path.join(self.working_folder, f"{subject}.npz")
        
        self.output_folder = os.path.join(root_folder, dst_folder)
        
        self.output_file = os.path.join(self.output_folder, f"{subject}_wavlet_{desc_level}.npz")
        
        self.desc_level = desc_level
        
        if os.path.exists(self.data_file):
            self._load_data_()
        
        else:
            raise FileNotFoundError(f"{self.data_file} not found")
        
        self._create_folder_()
        
    def _create_folder_(self) -> None:
        """Crea el folder de destino si no existe

        Raises:
            PermissionError: Si no se tiene permisos de escritura
        """
        
        if os.path.exists(self.output_folder):
            warnings.warn("The output path already exists, some data may be overwritten")
        
        else:
            try:
                os.mkdir(self.output_folder)
            except PermissionError:
                raise PermissionError(f"Permission denied to create {self.output_folder}")
    
    def _load_data_(self) -> None:
        """Carga los datos de EEG, los targets y los identificadores si existen
        Raises:
            ValueError: Si el EEG no tiene el formato correcto
        """
        
        print('Loading data...')
        
        data = np.load(self.data_file,
                       allow_pickle=True,
                       mmap_mode="r")
        
        self.EEG = data['EEG']
        self.targets = data['targets']
        
        try:
            self.identifiers = data["identifier"]
        except:
            self.identifiers = np.empty(shape=self.targets.shape)
        
        if len(self.EEG.shape) != 3:
            raise ValueError(f"{self.data} is not a 3D array")
    
    def apply(self, scale:Optional[bool]=True,
              save:Optional[bool]=True) -> Tuple:
        """Funcion encargada de aplicar la transformada de wavelet a los datos

        Args:
            scale (Optional[bool], optional): Si es verdadera aplicara un robust scaler despues de aplicar la transformada de wavelet. Defaults to True.
            save (Optional[bool], optional): Si es verdadera, guardara los datos y no retornara nada. Defaults to True.

        Returns:
            Tuple: Retornara los datos transformados y los targets si save es falsa
        """
        
        def _calc_shape_(num_samples:int, times:int):
            
            print(f"Calculating new shape for {num_samples} samples and {times} times")
            
            for _ in range(times):
                
                num_samples = math.ceil(num_samples / 2)
            
            return int(num_samples)

        print('Applying wavelet transform...')
        
        num_trails = self.EEG.shape[0]
        num_channels = self.EEG.shape[1]
        num_samples = self.EEG.shape[2]
        
        resampling_sahpe = _calc_shape_(num_samples, self.desc_level)
        
        self.wavelet_array = np.zeros(shape=(num_trails, num_channels, resampling_sahpe))
        
        print('Applying wavelet transform...')
        
        transform = dtcwt.Transform1d()
        
        for index, trail in enumerate(self.EEG):
            
            wavelet = transform.forward(trail.T, nlevels=self.desc_level)
            
            self.wavelet_array[index] = np.abs(wavelet.highpasses[self.desc_level - 1].T)
        
        if scale:
             
            scaler = MultivariateTransformer(RobustScaler(),
                                             flatten = False)

            self.wavelet_array = scaler.fit_transform(self.wavelet_array)
        
        if save:
            
            np.savez(self.output_file,
                     wavelet= self.wavelet_array,
                     targets= self.targets,
                     identifier= self.identifiers)
            
            print(f"with shape wavelet:{self.wavelet_array.shape} and {len(self.targets)} targets")

        else:
            
            return self.wavelet_array, self.targets

        del self.wavelet_array
        del self.EEG
        del self.targets
        
        gc.collect()


if __name__ == '__main__':
    
    wavelet = Wavelet(subject='MM05',
                      desc_level=6,
                      root_folder='Data',
                      src_folder='SplittedData',
                      dst_folder='WaveldData')
    
    wavelet.apply(scale=True, save=True)