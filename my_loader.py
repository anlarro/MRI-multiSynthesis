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
            # X = [nib.load(os.path.join(folder,f)).get_data() for f in sorted(os.listdir(folder))]
            # X = [np.swapaxes(np.swapaxes(d, 1, 2), 0, 1) for d in X]
            X_sitk = [sitk.ReadImage(os.path.join(folder, f)) for f in sorted(os.listdir(folder)) if '_reg' in f] #We load only the co-registered vols identified with _reg
            X = [sitk.GetArrayFromImage(vol_sitk) for vol_sitk in X_sitk]
            print('Loaded %d vols' % len(X))
            X = [X[i].astype('float32') for i in range(len(X))]
            self.vols[mod_name] = X
            self.vols_sitk[mod_name] = X_sitk
        self.num_vols = len(X)
        self.patient_names = [f.split('-')[0] for f in sorted(os.listdir(folder))]

    def padVolume(self,volume, padding, default_pixel_value=0):
        padded = sitk.ConstantPad(volume, [4, 4, 0], [2, 2, 0])
        return padded

    def get(self, modality, ids):
        shapes = [np.mean(np.shape(self.vols[modality][i])[1:3]) for i in ids]
        dim = int(4 * np.ceil(np.max(shapes) / 4)) #make all 2D dimensions divisble by 4
        data_ids = []
        for i in ids:
            shape=np.shape(self.vols[modality][i])
            data_ids.append(np.pad(
                self.vols[modality][i],
                ((0,0), (int(np.floor(abs(dim-shape[1])/2)),
                         int(np.ceil(abs(dim-shape[1])/2))),
                 (int(np.floor(abs(dim-shape[2])/2)),
                  int(np.ceil(abs(dim-shape[2])/2)))),
                 mode = 'constant'))
        data_ids_ar = np.concatenate(data_ids)
        if len(data_ids_ar.shape) < 4:
            data_ids_ar = np.expand_dims(data_ids_ar, axis=1)
        return data_ids_ar

    # def normalize(self, ids):
    #     for i, x in enumerate(X):
    #         X[i] = X[i] / np.mean(x)