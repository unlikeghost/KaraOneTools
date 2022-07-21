import os
import glob 
import shutil
import tarfile
import requests
from typing import Optional


class Downloader:
    """Clase para descargar y extraer los archivos de KaraOne
    """
    
    def __init__(self, url:Optional[str]="http://www.cs.toronto.edu/~complingweb/data/karaOne",
                 download_path:Optional[str]='Data'):
        """Constructor de la clase Downloader

        Args:
            url (Optional[str], optional): Url de donde estan los datasets de KaraOne. Defaults to "http://www.cs.toronto.edu/~complingweb/data/karaOne".
            download_path (Optional[str], optional): Path donde se guardaran los datos. Defaults to 'Data'.

        Raises:
            PermissionError: Si no se tiene permisos para crear el directorio o eliminar directorios.
        """
        
        self.url = url
        self.downlad_path = download_path
        try:
            os.mkdir(download_path)
        
        except FileExistsError:
            print("Directory already exists, skipping creation")
        
        except PermissionError:
            raise PermissionError("Permission denied, exiting")
    
    def _create_folder(self, folder:str):
        
        if not os.path.exists(folder):
            
            try:
                os.mkdir(folder)
                
            except PermissionError:
                raise PermissionError("Permission denied, exiting")
            
            else:
                return True
        
        else:
            try:
                shutil.rmtree(folder)
                
            except PermissionError:
                raise PermissionError("Permission denied, exiting")
            
            else:
                os.mkdir(folder)
                return True
            
    def downlad(self, subject:str, output_path:Optional[str]='RawDataCompressed'):
        """Metodo para descargar los archivos de KaraOne

        Args:
            subject (str): Sujeto a descargar, nombrado como en KaraOne
            output_path (Optional[str], optional): Folder donde se descargaran los daros. Defaults to 'RawDataCompressed'.
        """
        
        output_path = os.path.join(self.downlad_path, output_path)
        
        if self._create_folder(output_path):
            
            try:
                
                print(f"Downloading {subject}")
                 
                with requests.get(f"{self.url}/{subject}.tar.bz2",
                                stream=True, allow_redirects=True) as response:
                    
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=2048):
                        open(os.path.join(output_path, f"{subject}.tar.bz2"), "ab").write(chunk)
                    
                response.close()
            
            except requests.exceptions.HTTPError:
                print(f"Invalid URL: {self.url}{subject}.tar.bz2 maybe the subject {subject} is not available or dosn't exist")
            
            else:
                print(f"Downloaded {subject}")
    
    def extract(self, subject:str, src_folder:Optional[str]='RawDataCompressed',
                dst_folder:Optional[str]='RawDataExtracted'):
        """Metodo para extraer los archivos de KaraOne descargados y moverlos a folder root

        Args:
            subject (str): Sujetos a extraer, nombrado como en KaraOne
            src_folder (Optional[str], optional): Path donde se guardaron los archivos comprimidos. Defaults to 'RawDataCompressed'.
            dst_folder (Optional[str], optional): Path donde se guardaran los archivos descomprimidos. Defaults to 'RawDataExtracted'.

        Raises:
            tarfile.ReadError: Error de lectura en el archivo comprimido.
            FileNotFoundError: Archivo no encontrado.
            PermissionError: Permisos para crear el directorio, eliminar directorios o modificar archivos.
        """
        
        output_path = os.path.join(self.downlad_path, dst_folder)
        
        src_path = os.path.join(self.downlad_path, src_folder, f"{subject}.tar.bz2")
        
        create = self._create_folder(output_path)
        
        if create and os.path.exists(src_path):
            
            print(f"Extracting {subject}")
            
            try:
                with tarfile.open(src_path) as tar:
                    tar.extractall(os.path.join(output_path, subject))
                    
                tar.close()
                
                print(f"Done extracting {subject}")
            
            except tarfile.ReadError:
                raise tarfile.ReadError(f"Invalid tar file: {subject}.tar.bz2")
            
            except FileNotFoundError:
                raise FileNotFoundError("File not found, exiting")
            
            else:
                
                folders = [folder[0] for folder in os.walk(os.path.join(output_path,
                                                                        subject)) if len(folder[2])]
                
                root_folders = list(filter(lambda folder: folder.split("\\")[-1] == subject, folders))
                
                files = glob.glob(os.path.join(root_folders[0], "*"))
                
                movefiles = lambda file: shutil.move(file, os.path.join(output_path, subject))
                
                try:
                    print("Moving files...")
                    
                    list(map(movefiles, files))
                
                except PermissionError:
                    raise PermissionError("Permission denied, exiting")
                
                else:
                    print(f"Done moving files for {subject}")
                
        else:
            
            print(f"{subject} not found, skipping extraction")
   
            
if __name__ == '__main__':
    
    test = Downloader()
    # test.downlad("MM05")
    test.extract("MM05")