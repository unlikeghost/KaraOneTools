import numpy as np
from typing import Tuple


class DataAugmentation:
    """Clase para aumentar datos de EEG, basada en los metodos de jitter y scaling
    
    Acerca de las tecnicas:
        scaling: https://arxiv.org/pdf/1706.00527.pdf
        jitter:  https://arxiv.org/pdf/1706.00527.pdf
    """
    
    def __init__(self, x:np.ndarray, y:np.ndarray, augmentation_factor:int):
        """Constructor de la clase
        
        Args:
            x (np.ndarray): EEG array de 3 dimensiones (epochs, channels, time)
            y (np.ndarray): labels array de 1 dimension (label)
            augmentation_factor (int): Cuantas veces se aumentaran los datos

        Raises:
            ValueError: Si el EEG no son de 3 dimensiones
            ValueError: Si x y y no son del mismo tamaño
            ValueError: si el factor de aumentacion no es mayor a 1
        """
        if len(x.shape) != 3:
            raise ValueError("x must be a 3D array")
        
        if len(y) != len(x):
            raise ValueError("x and y must have the same length")
        
        if augmentation_factor < 1:
            raise ValueError("augmentation_factor must be greater than 1")
        
        self.augmentation_factor = augmentation_factor
        
        self.eeg = x
        
        self.labels = y
        
        self.current = x.shape[0]
        
        output_shape = ((self.augmentation_factor * x.shape[0]) + x.shape[0])
        
        self.x_output_array = np.zeros(shape=(output_shape,
                                              x.shape[1],
                                              x.shape[2])
                                       )
        
        self.y_output_array = np.empty(shape = (output_shape),
                                       dtype = np.object_)
        
        self.identifier_array = np.empty(shape = (output_shape),
                                       dtype = np.object_)
    
    def jitter(self, low_sigma:float, high_sigma:float) -> Tuple:
        """Toddo acerca de Jitter: https://arxiv.org/pdf/1706.00527.pdf
        
        Se creara un rango dados tus valores de low y high, y se creara un rango de valores para jitter
        Args:
            low_sigma (float): Sigma menor para jitter
            high_sigma (float): Sigma mayor para jitter

        Returns:
            np.ndarray: 3D array de EEG con los datos aumentados
            np.ndarray: 1D array de labels con los datos aumentados
            np.ndarray: 1D array de identificadores con los datos aumentados (Original, jitter__sigma_SigmaLevel__target_LabelName, ...)
        """
        current = self.current
                
        sigmas = np.linspace(low_sigma, high_sigma, self.augmentation_factor)
        
        type_augmented = ['jitter__sigma_{:.1e}'.format(sigma) for sigma in sigmas]
        
        keys = range(len(sigmas))
        
        dict_identifier = dict(zip(keys, type_augmented))
        
        self.x_output_array[:self.eeg.shape[0], :, :] = self.eeg
        
        self.y_output_array[:self.eeg.shape[0]] = self.labels
        
        self.identifier_array[:self.eeg.shape[0]] = ['original'] * self.eeg.shape[0]
        
        for index, sigma in enumerate(sigmas):
            for array, target in zip(self.eeg, self.labels):
                
                temp_array = array + np.random.normal(loc = 0.,
                                                      scale = sigma,
                                                      size = array.shape)
                
                self.x_output_array[current, :, :]  = temp_array
                
                self.y_output_array[current] = target
                
                self.identifier_array[current] = f'{dict_identifier[index]}__target_{target}'
                
                current += 1
        
        return self.x_output_array, self.y_output_array, self.identifier_array
    
    def scaling(self, low_sigma, high_sigma):
        """Toddo acerca de Scaling: https://arxiv.org/pdf/1706.00527.pdf
        
        Se creara un rango dados tus valores de low y high, y se creara un rango de valores para jitter
        Args:
            low_sigma (float): Sigma menor para jitter
            high_sigma (float): Sigma mayor para jitter


        Returns:
            np.ndarray: 3D array de EEG con los datos aumentados
            np.ndarray: 1D array de labels con los datos aumentados
            np.ndarray: 1D array de identificadores con los datos aumentados (Original, jitter_1, jitter_2, ...)
        """
        current = self.current
        
        sigmas = np.linspace(low_sigma, high_sigma, num=self.augmentation_factor)
        
        type_augmented = ['scaling__sigma_{:.1e}'.format(sigma) for sigma in sigmas]
        
        keys = range(len(sigmas))
        
        dict_identifier = dict(zip(keys, type_augmented))
        
        self.x_output_array[:self.eeg.shape[0], :, :] = self.eeg
        
        self.y_output_array[:self.eeg.shape[0]] = self.labels
        
        self.identifier_array[:self.eeg.shape[0]] = ['original'] * self.eeg.shape[0]
        
        for index, sigma in enumerate(sigmas):
            for array, target in zip(self.eeg, self.labels):
                
                array = np.expand_dims(array, axis=0)
                
                factor = np.random.normal(loc=1.,
                                          scale=sigma,
                                          size=(1,
                                                self.eeg.shape[2])
                                          )
                
                self.x_output_array[current, :, :]  = np.multiply(array,
                                                                  factor[ :,np.newaxis,:])
                
                self.y_output_array[current] = target
                                
                self.identifier_array[current] = f'{dict_identifier[index]}__target_{target}'

                current += 1
                
        return self.x_output_array, self.y_output_array, self.identifier_array

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    eeg_fake = np.random.randint(0, 100, size=(3, 10, 50))
    
    labels = np.array(list(range(0, 3)))
    
    augmentation_factor = 3
    
    DA = DataAugmentation(eeg_fake, labels, augmentation_factor)
    
    x_jitter, labels_jitter, ident_jitter = DA.jitter(low_sigma=5, high_sigma=6)
    
    plt.plot(x_jitter[0, 0, 0:5], label=f'{ident_jitter[0]}__target_{labels_jitter[0]}')
    plt.plot(x_jitter[3, 0, 0:5], label=ident_jitter[3])
    
    plt.title('EEG con jitter (Ejemplo)')
    plt.legend()
    plt.show()
    
    
    x_scale, labels_scale, ident_scale = DA.scaling(low_sigma=1, high_sigma=2)
    
    plt.plot(x_scale[0, 0, 0:5], label=f'{ident_scale[0]}__target_{labels_scale[0]}')
    plt.plot(x_scale[3, 0, 0:5], label=ident_scale[3])
    
    plt.title('EEG con Scaling (Ejemplo)')
    plt.legend()
    plt.show()