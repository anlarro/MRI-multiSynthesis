import os
import numpy as np

import SimpleITK as sitk

class Data(object):

    def __init__(self, train_folder = None, valid_folder = None, test_folder = None):
        self.folders={}
        self.folders['train'] = train_folder
        self.folders['valid'] = valid_folder
        self.folders['test'] = test_folder
        self.max_size = (0, 0)

        if test_folder is not None:
            folder_mod = os.path.join(test_folder, os.listdir(test_folder)[0]);
            self.test_names = [f.split('_')[0] for f in sorted(os.listdir(folder_mod)) if '_reg' in f]
            self.total_vols = len(self.test_names)
        else:
            folder_mod = os.path.join(train_folder, os.listdir(train_folder)[0]);
            train_files = [f for f in sorted(os.listdir(folder_mod)) if '_reg' in f]
            self.total_vols = len(train_files)

    def generate_batches(self, input_modalities, output_modalities = None, num_emb = None, batch_size = 16, mode='train'):
        #yields batches of data, to use with fit_generator
        folder=self.folders[mode]
        vol_num = 0
        X = self.readCase(folder, vol_num, input_modalities)
        Y = self.readCase(folder, vol_num, output_modalities)
        last_vol = X
        vol_num+=1
        while True: #loop indefinitely (generator)
            if X[0].shape[0] <= last_vol[0].shape[0]:
                X2 = self.readCase(folder, vol_num, input_modalities)
                Y2 = self.readCase(folder, vol_num, output_modalities)
                X = [np.vstack((X[i],X2[i])) for i in range(len(X))] #read the next volume, we always keep >1 volume_size in X
                Y = [np.vstack((Y[i], Y2[i])) for i in range(len(Y))]  # read the next volume, we always keep >1 volume_size in X
                last_vol = X2
                vol_num += 1
                if vol_num == self.total_vols and mode != 'test': # we keep feeding the generator from the fist volume
                    vol_num = 0
                    X = self.readCase(folder, vol_num, input_modalities)
                    Y = self.readCase(folder, vol_num, output_modalities)
                    last_vol = X
                    vol_num += 1

            # we always pop() batch_size from the beginning of X
            if X[0].shape[0] >= batch_size:
                batch_in = [X[i][:batch_size].copy() for i in range(len(X))]
                X = [np.delete(X[i], np.array(range(batch_size)), 0) for i in range(len(X))]

                batch_out = [Y[i][:batch_size].copy() for i in range(len(Y))]
                Y = [np.delete(Y[i], np.array(range(batch_size)), 0) for i in range(len(Y))]
            else:   #this will occur in test mode when we reach the final slices
                batch_in = X
                batch_out = Y
                X = []
                Y = []

            if len(batch_in[0].shape) < 4:
                batch_in = [np.expand_dims(batch_in[i], axis=1) for i in range(len(batch_in))]
            if len(batch_out[0].shape) < 4:
                batch_out = [np.expand_dims(batch_out[i], axis=1) for i in range(len(batch_out))]

            # there's 1 output per embedding plus 1 output for the total variance embedding
            batch_out = [batch_out[m] for m in range(len(batch_out)) for i in range(num_emb)]
            batch_shape = (batch_out[0].shape[0], 1, batch_out[0].shape[2], batch_out[0].shape[3])
            if len(input_modalities) > 1:
                batch_out += [np.zeros(shape=batch_shape) for i in range(2)]
            yield(batch_in, batch_out)

    def readCase(self, folder, vol_num, modalities):
        array_volume = []
        for mod_name in modalities:
            folder_mod = os.path.join(folder, mod_name)
            files = [f for f in sorted(os.listdir(folder_mod)) if '_reg' in f]  # We load only the co-registered vols identified with _reg
            f = files[vol_num]
            volume = sitk.ReadImage(os.path.join(folder_mod, f))
            volume_padded = self.padVolume(volume, self.target_size)  # With padding we ensure that all slices across volumes are the same
            array_volume.append(sitk.GetArrayFromImage(volume_padded).astype('float32'))
        return array_volume

    def padVolume(self,volume, size, default_pixel_value=0):
        shape = volume.GetSize()

        padded = sitk.ConstantPad(volume,
                                  [int(np.floor(abs(size[0]-shape[0])/2)), int(np.floor(abs(size[1]-shape[1])/2)), 0],
                                  [int(np.ceil(abs(size[0]-shape[0])/2)), int(np.ceil(abs(size[1]-shape[1])/2)), 0],
                                  default_pixel_value)
        return padded

    def countImages(self, input_modalities, mode = 'train'):  #loads all volumes, to use with model.fit()
        mod_name = input_modalities[0]
        folder = self.folders[mode]
        folder_mod = os.path.join(folder, mod_name)
        files = [f for f in sorted(os.listdir(folder_mod)) if '_reg' in f]
        num_images = 0
        num_vols = 0
        for f in files:
            volume = sitk.ReadImage(os.path.join(folder_mod, f))
            num_images+=volume.GetSize()[2]
            num_vols+=1
            if volume.GetSize()[:2] > self.max_size:
                self.max_size = volume.GetSize()[:2]
            self.target_size = [int(4 * np.ceil((self.max_size[i] + 4) / 4)) for i in range(2)]
            self.vol_shape = self.target_size[::-1]
        return num_images, num_vols