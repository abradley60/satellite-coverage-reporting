from landsatxplore.earthexplorer import EarthExplorer
from sentinelsat import SentinelAPI
import os
import tarfile
import zipfile

class LandsatDownloader():
    
    def __init__(self, user, password, email=''):
        self.user = user
        self.email = email
        self.api = EarthExplorer(user, password)

    def logout(self):
        self.api.logout()

    def download(self, product, data_dir='data'):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        final_path = os.path.join(data_dir,product)
        print(final_path)
        if not os.path.exists(f'{final_path}'):
            if not os.path.exists(f'{final_path}.tar'):
                print(f'downloading product to - {final_path}')
                self.api.download(product, output_dir=data_dir)
        if not os.path.exists(f'{final_path}'):
            #unzip to folder
            import tarfile
            # open file
            with tarfile.open(f'{final_path}.tar') as f:
                # extracting file
                f.extractall(final_path)
        else:
            print(f'product already exists - {final_path}')

class S2Downloader():
    def __init__(self, user, password, email=''):
        self.user = user
        self.email = email
        self.api = SentinelAPI(user, password)

    def download(self, product, data_dir='data'):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        final_path = os.path.join(data_dir,product)
        print(final_path)
        if not os.path.exists(f'{final_path}'):
            if not os.path.exists(f'{final_path}.zip'):
                print(f'downloading product to - {final_path}')
                self.api.download(product, directory_path=data_dir) #nodefilter=path_filter)
        if not os.path.exists(f'{final_path}'):
            import zipfile
            with zipfile.ZipFile(f'{data_dir}/{product}.zip', 'r') as zip_ref:
                zip_ref.extractall(final_path)
        else:
            print(f'product already exists - {final_path}')
