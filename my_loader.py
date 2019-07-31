import os
import numpy as np

import SimpleITK as sitk

class Data(object):

    def __init__(self, train_folder = None, valid_folder = None, test_folder = None):
        self.folders={}
        self.folders['train'] = train_folder[:-1] if train_folder.endswith('/') else train_folder
        self.folders['valid'] = (valid_folder[:-1] if valid_folder.endswith('/') else valid_folder)
        self.folders['test'] = (test_folder[:-1] if test_folder.endswith('/') else test_folder)

        if test_folder is not None:
            self.test_names = [f.split('_')[0] for f in sorted(os.listdir(test_folder)) if '_reg' in f]

    def generate_batches(self,  modalities, batch_size, mode='train'): #loads batches of data, to use with fit_generator
        folder=self.folders[mode]
        vol_num = 0
        slice_num = 0
        while True:
            # initialize our batches of images and labels
            batch_in = []
            batch_out = []
            X = []
            # yield batch_size slices
            for mod_name in modalities:
                folder_mod = os.path.join(folder, mod_name)
                files = [f for f in sorted(os.listdir(folder_mod)) if '_reg' in f] # We load only the co-registered vols identified with _reg
                f = files[vol_num]
                volume = sitk.ReadImage(os.path.join(folder_mod, f))
                target_size = [int(32 * np.ceil((volume.GetSize()[i]+32) / 32)) for i in range(2)]  # make target 2D size to be divisible by 32
                volume_padded = self.padVolume(volume, target_size)
                X.append(sitk.GetArrayFromImage(volume_padded).astype('float32'))

            slices_per_volume = X[0].shape[0]
            if (slice_num + batch_size) < slices_per_volume:
                batch_in = [X[i][slice_num:slice_num + batch_size] for i in range(len(X))]
                slice_num += batch_size
                vol_num += 1

            if vol_num > len(os.listdir(folder))-1: # we reached the last file, reset a re-read the first file
                vol_num=0
                volume = sitk.ReadImage(os.path.join(folder, f))
                target_size = [int(32 * np.ceil((volume.GetSize()[i] + 32) / 32)) for i in range(2)]  # make target 2D size to be divisible by 32, to ensure that all slices are of the same size
                volume_padded = self.padVolume(volume, target_size)
                X = sitk.GetArrayFromImage(volume_padded).astype('float32')

                # if we are testing we should now break from our
                # loop to ensure we don't continue to fill up the
                # batch from samples at the beginning of the file
                if mode == "test":
                    break

                data_ids_ar = np.concatenate(data_ids)
                if len(data_ids_ar.shape) < 4:
                    data_ids_ar = np.expand_dims(data_ids_ar, axis=1)

                train_in = [data.get(mod, ids_train) for mod in input_modalities]

                # there's 1 output per embedding plus 1 output for the total variance embedding
                train_out = [data.get(mod, ids_train) for mod in output_modalities for i in range(m.num_emb)]

                train_shape = (train_out[0].shape[0], 1, train_out[0].shape[2], train_out[0].shape[3])

                if len(input_modalities) > 1:
                    train_out += [np.zeros(shape=train_shape) for i in range(2)]

                # update our corresponding batches lists
                batch_in.append(inputs)
                batch_out.append(outputs)

                yield(batch_in, batch_out)

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