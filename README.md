# KaraOne Tools

KaraOne Tools es una librería de Python que nos permitirá trabajar con el dataset de [KaraOne EEG](http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html) de una maneta más sencilla

## Instalación

Clona este repositorio e instala las dependencias
```bash
git clone https://github.com/unlikeghost/karaone.git
```

```bash
pip install -r requirements.txt
```

## Uso

### Descarga del dataset
```python
from KaraOne import Downloader

test = Downloader()
test.downlad("MM05")
test.extract("MM05")
```

### Aumento de datos
```python
import matplotlib.pyplot as plt
from KaraOne import DataAugmentation    

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
```

### Separación de datos
```python
from KaraOne import SplitData
test = SplitData(subject='MM05',
                     type_action='thinking_inds',
                     ignore_channels=['VEO', 'HEO','EKG', 'EMG','Trigger'],
                     keep_channels=['T7', 'C5', 'C3', 'CP5', 'CP3', 'CP1', 'P3', 'C4', 'FC6', 'FT8'],
                     root_folder='Data',
                     src_folder='RawDataExtracted',
                     dst_folder='SplittedData')

test.split(duration_time=4400, save=True)
```

### Transformada de wavelet
```python
from KaraOne import Downloader

wavelet = Wavelet(subject='MM05',
                  desc_level=6,
                  root_folder='Data',
                  src_folder='SplittedData',
                  dst_folder='WaveldData')

wavelet.apply(scale=True, save=True)
```

## Autor 
* Jesus Alan Hernadnez Galvan [a329691@uach.mx](email)