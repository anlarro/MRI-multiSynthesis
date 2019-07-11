import os
import math
import itertools

from model import Multimodel
from my_loader import Data
import SimpleITK as sitk
import numpy as np

folder = '/mnt/D8D413E4D413C422/I3M/Imagenes/Oasis/data-reduced'

training_inputs = ['T1','T2','T2-FLAIR']
training_outputs = ['T1','T2','T2-FLAIR']

data = Data(folder, training_inputs)
data.load()

#build model
weights = {m:1.0 for m in training_outputs}
weights['concat']=1.0
m = Multimodel(training_inputs, training_outputs, weights, 16, 1, True, 'max', True, True, data.vol_shape)
m.build()
#load weights
m.model.load_weights('my_weights.h5')


output_folder = os.path.join(folder,'outputs')

try:
    os.mkdir(output_folder)
except OSError:
    print ("Output folder already exist")

trainFrac = math.floor(data.num_vols*0.7)
valFrac = math.floor(data.num_vols*0.1)
testFrac = math.floor(data.num_vols*0.2)
ids_test = range(trainFrac+valFrac,trainFrac+valFrac+testFrac)

inputs = []
for n in range(1,len(training_outputs)):
    inputs.extend(list(itertools.combinations(training_inputs, n)))

outputs = []
for i in inputs:
    outputs.append(tuple(set(i).symmetric_difference(training_inputs)))

for vol_num in ids_test:
    print('testing model on volume ' + str(vol_num) + '...')
    for i in range(len(inputs)):
        input_modalities = list(inputs[i])
        output_modalities = list(outputs[i])
        for o in range(len(output_modalities)):
            X = [data.get(mod, [vol_num]) for mod in input_modalities]
            partial_model = m.get_partial_model(input_modalities, output_modalities[o])
            Z = partial_model.predict(X)
                # there's 1 output per embedding plus 1 output for the total variance embedding (that's why we iterate as follows)

            index = len(input_modalities)
            padded_size = np.squeeze(Z[index]).shape
            no_padded_size = data.vols_sitk[input_modalities[0]][vol_num].GetSize()[::-1]
            Z_sitk = sitk.GetImageFromArray(np.squeeze(Z[index])[:,
                                            int(np.floor((padded_size[1] - no_padded_size[1]) / 2)):
                                            no_padded_size[1] + int(
                                            np.floor((padded_size[1] - no_padded_size[1]) / 2)),
                                            int(np.floor((padded_size[2] - no_padded_size[2]) / 2)):
                                            no_padded_size[2] + int(
                                            np.floor((padded_size[2] - no_padded_size[2]) / 2))
                                            ])
            Z_sitk.CopyInformation(data.vols_sitk[input_modalities[0]][vol_num])
            sitk.WriteImage(Z_sitk, os.path.join(output_folder, data.patient_names[vol_num] +
                                                '_in_' + "--".join(input_modalities) + '_out_' + output_modalities[o] + '.nii.gz'))

