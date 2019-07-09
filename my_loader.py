import os
import numpy as np

import SimpleITK as sitk

class Data(object):
    '''
    Example usage of the class:
    data = Data('./data/ISLES', modalities_to_load=['T1','T2'])
    data.load()
    vols_0_1_2 = data.select_for_ids('T1', [0, 1, 2])
    '''
    def __init__(self, folder, modalities):
        self.folder = folder[:-1] if folder.endswith('/') else folder
        self.modalities = modalities
        self.vols = {}
        self.vols_sitk = {}

    def load(self):
        for mod_name in self.modalities:
            print('Loading ' + mod_name)
            folder=os.path.join(self.folder,mod_name)
            X_sitk = [sitk.ReadImage(os.path.join(folder, f)) for f in sorted(os.listdir(folder)) if '_reg' in f] #We load only the co-registered vols identified with _reg
            target_size = [int(4 * np.ceil((X_sitk[0].GetSize()[i]+4) / 4)) for i in range(2)] #make target 2D size to be divisible by 8
            X_sitk_padded = [self.padVolume(X,target_size) for X in X_sitk]
            X = [sitk.GetArrayFromImage(vol_sitk) for vol_sitk in X_sitk_padded]
            print('Loaded %d vols' % len(X))
            X = [X[i].astype('float32') for i in range(len(X))]
            self.vols[mod_name] = X
            self.vols_sitk[mod_name] = X_sitk   #we keep the original sitk volume without padding
        self.num_vols = len(X)
        self.patient_names = [f.split('-')[0] for f in sorted(os.listdir(folder)) if '_reg' in f]

    def padVolume(self,volume, size, default_pixel_value=0):
        shape = volume.GetSize()

        padded = sitk.ConstantPad(volume,
                                  [int(np.floor(abs(size[0]-shape[0])/2)), int(np.floor(abs(size[1]-shape[1])/2)), 0],
                                  [int(np.ceil(abs(size[0]-shape[0])/2)), int(np.ceil(abs(size[1]-shape[1])/2)), 0],
                                  default_pixel_value)
        return padded

    def get(self, modality, ids):
        data_ids = [self.vols[modality][i] for i in ids]

        data_ids_ar = np.concatenate(data_ids)
        if len(data_ids_ar.shape) < 4:
            data_ids_ar = np.expand_dims(data_ids_ar, axis=1)
        return data_ids_ar

    # def normalize(self, ids):
    #     for i, x in enumerate(X):
    #         X[i] = X[i] / np.mean(x)