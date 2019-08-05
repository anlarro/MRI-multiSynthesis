import os
import numpy as np

import SimpleITK as sitk

class Data(object):

    def __init__(self, train_folder = None, valid_folder = None, test_folder = None):
        self.folders={}
        self.folders['train'] = train_folder
        self.folders['valid'] = valid_folder
        self.folders['test'] = test_folder

        if test_folder is not None:
            folder_mod = os.path.join(test_folder, os.listdir(test_folder)[0]);
            self.test_names = [f.split('_')[0] for f in sorted(os.listdir(folder_mod)) if '_reg' in f]
            self.total_vols = len(self.test_names)
        else:
            folder_mod = os.path.join(train_folder, os.listdir(train_folder)[0]);
            train_files = [f for f in sorted(os.listdir(folder_mod)) if '_reg' in f]
            self.total_vols = len(train_files)

    def generate_batches(self, modalities, num_emb = None, batch_size = 16, mode='train'):
        #yields batches of data, to use with fit_generator
        folder=self.folders[mode]
        vol_num = 0
        X = readVolume(folder, vol_num, modalities)
        last_vol = X
        vol_num+=1
        while True: #loop indefinitely (generator)

            if X[0].shape[0] <= last_vol[0].shape[0]:
                X2 = readVolume(folder, vol_num, modalities)
                X = [np.vstack((X[i],X2[i])) for i in range(len(X))] #read the next volume, we always keep >1 volume_size in X
                last_vol = X2
                vol_num += 1
                if vol_num == self.total_vols and mode != 'test': # we keep feeding the generator from the fist volume
                    vol_num = 0
                    X = readVolume(folder, vol_num, modalities)
                    last_vol = X
                    vol_num += 1

            # we always pop() batch_size from the beginning of X
            if X[0].shape[0] >= batch_size:
                batch = [X[i][:batch_size].copy() for i in range(len(X))]
                X = [np.delete(X[i], np.array(range(batch_size)), 0) for i in range(len(X))]
            else:   #this will occur in test mode when we reach the final slices
                batch = X
                X = []

            if len(batch[0].shape) < 4:
                batch = [np.expand_dims(batch[i], axis=1) for i in range(len(batch))]

            if num_emb is not None:
                # there's 1 output per embedding plus 1 output for the total variance embedding
                batch = [batch[m] for m in range(len(batch)) for i in range(num_emb)]
#
#             train_shape = (train_out[0].shape[0], 1, train_out[0].shape[2], train_out[0].shape[3])
#
#             if len(input_modalities) > 1:
#                 train_out += [np.zeros(shape=train_shape) for i in range(2)]
# ####################
            yield(batch)

    def readVolume(self, folder, vol_num, modalities):
        array_volume = []
        for mod_name in modalities:
            folder_mod = os.path.join(folder, mod_name)
            files = [f for f in sorted(os.listdir(folder_mod)) if '_reg' in f]  # We load only the co-registered vols identified with _reg
            f = files[vol_num]
            volume = sitk.ReadImage(os.path.join(folder_mod, f))
            target_size = [int(32 * np.ceil((volume.GetSize()[i] + 32) / 32)) for i in
                           range(2)]  # make target 2D size to be divisible by 32
            volume_padded = self.padVolume(volume,
                                           target_size)  # With padding we ensure that all slices across volumes are the same
            array_volume.append(sitk.GetArrayFromImage(volume_padded).astype('float32'))
        return array_volume

    def padVolume(self,volume, size, default_pixel_value=0):
        shape = volume.GetSize()

        padded = sitk.ConstantPad(volume,
                                  [int(np.floor(abs(size[0]-shape[0])/2)), int(np.floor(abs(size[1]-shape[1])/2)), 0],
                                  [int(np.ceil(abs(size[0]-shape[0])/2)), int(np.ceil(abs(size[1]-shape[1])/2)), 0],
                                  default_pixel_value)
        return padded

    def load(self):    #loads all volumes, to use with model.fit()
        for mod_name in self.modalities:
            print('Loading ' + mod_name)
            folder=os.path.join(self.folder,mod_name)
            X_sitk = [sitk.ReadImage(os.path.join(folder, f)) for f in sorted(os.listdir(folder)) if '_reg' in f] #We load only the co-registered vols identified with _reg
            target_size = [int(32 * np.ceil((X_sitk[0].GetSize()[i]+32) / 32)) for i in range(2)] #make target 2D size to be divisible by 32, to ensure that all slices are of the same size
            X_sitk_padded = [self.padVolume(X,target_size) for X in X_sitk]
            X = [sitk.GetArrayFromImage(vol_sitk) for vol_sitk in X_sitk_padded]
            print('Loaded %d vols' % len(X))
            X = [X[i].astype('float32') for i in range(len(X))]
            self.vols[mod_name] = X
            self.vols_sitk[mod_name] = X_sitk   #we keep the original sitk volume without padding
        self.vol_shape = target_size[::-1]
        self.num_vols = len(X)

    def get(self, modality, ids):  #get volumes, to use after load, to use with model.fit()
        data_ids = [self.vols[modality][i] for i in ids]

        data_ids_ar = np.concatenate(data_ids)
        if len(data_ids_ar.shape) < 4:
            data_ids_ar = np.expand_dims(data_ids_ar, axis=1)
        return data_ids_ar