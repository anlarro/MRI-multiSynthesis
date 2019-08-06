import os
import math
import itertools

from model import Multimodel
from my_loader import Data
import SimpleITK as sitk
import numpy as np

folder = '/mnt/D8D413E4D413C422/I3M/Imagenes/Oasis/data-reduced'

#The model was trained with all possible inputs and outputs, so we have one sigle model.
# At testing time the required output is not included as input. For example: output: T2-FLAIR - inputs: T1-T2
training_inputs = ['T1','T2','T2-FLAIR']
training_outputs = ['T1','T2','T2-FLAIR']

data = Data(train_folder = folder+'/Training',test_folder = folder+'/Testing')
print("Counting total number of images...")
# we need to count training images again in order to have the correct data.vol_shape
num_train_images, _ = data.countImages(training_inputs, mode = 'train')
num_test_images, test_vols = data.countImages(training_inputs, mode = 'test')

#build model
weights = {m:1.0 for m in training_outputs}
weights['concat']=1.0
m = Multimodel(training_inputs, training_outputs, weights, 16, 1, True, 'max', True, True, data.vol_shape)
m.build(test=True)
#load weights
m.model.load_weights('my_weights.h5')

output_folder = os.path.join(folder,'outputs')

try:
    os.mkdir(output_folder)
except OSError:
    print ("Output folder already exist")

inputs = []
for n in range(1,len(training_outputs)):
    inputs.extend(list(itertools.combinations(training_inputs, n)))

outputs = []
for i in inputs:
    outputs.append(tuple(set(i).symmetric_difference(training_inputs)))

BATCH_SIZE = 1
for vol_num in range(test_vols):
    print('testing model on volume ' + str(vol_num) + '...')
    _, current_sitkVol = data.readCase(data.folders['test'], vol_num, [training_inputs[0]])
    slices_per_volume = current_sitkVol.GetSize()[2]
    for i in range(len(inputs)):
        input_modalities = list(inputs[i])
        output_modalities = list(outputs[i])
        for o in output_modalities:
            testGen = data.generate_batches(input_modalities, batch_size=BATCH_SIZE, mode='test')
            partial_model = m.get_partial_model(input_modalities, o)
            Z = partial_model.predict_generator(testGen, val_samples = slices_per_volume)

            index = len(input_modalities)
            padded_size = np.squeeze(Z[index]).shape
            no_padded_size = current_sitkVol.GetSize()[::-1]
            Z_sitk = sitk.GetImageFromArray(np.squeeze(Z[index])[:,
                                            int(np.floor((padded_size[1] - no_padded_size[1]) / 2)):
                                            no_padded_size[1] + int(
                                            np.floor((padded_size[1] - no_padded_size[1]) / 2)),
                                            int(np.floor((padded_size[2] - no_padded_size[2]) / 2)):
                                            no_padded_size[2] + int(
                                            np.floor((padded_size[2] - no_padded_size[2]) / 2))
                                            ])
            Z_sitk.CopyInformation(current_sitkVol)
            sitk.WriteImage(Z_sitk, os.path.join(output_folder, data.test_names[vol_num] +
                                                '_in_' + "--".join(input_modalities) + '_out_' + o + '.nii.gz'))

