import os
import math

from model import Multimodel
from my_loader import Data
import SimpleITK as sitk
import numpy as np

input_modalities = ['PD']
output_modalities = ['T2']
folder = '/mnt/D8D413E4D413C422/I3M/Imagenes/IXI/IXI-reduced'

data = Data(folder, input_modalities + output_modalities)
data.load()

output_folder = os.path.join(folder,'outputs')

#build model
weights = {m:1.0 for m in output_modalities}
weights['concat']=1.0
m = Multimodel(input_modalities, output_modalities, weights, 16, 1, False, 'max', True, True)
m.build()
#load weights
m.model.load_weights('my_weights.h5')

try:
    os.mkdir(output_folder)
except OSError:
    print ("Output folder already exist")

trainFrac = math.floor(data.num_vols*0.7)
valFrac = math.floor(data.num_vols*0.1)
testFrac = math.floor(data.num_vols*0.2)
ids_test = range(trainFrac+valFrac,trainFrac+valFrac+testFrac)

for vol_num in ids_test:
    print('testing model on volume ' + str(vol_num) + '...')
    X = [data.get(mod, [vol_num]) for mod in input_modalities]
    Z = m.model.predict(X)
    # there's 1 output per embedding plus 1 output for the total variance embedding (that's why we iterate as follows)
    for i in range(np.shape(Z)[0]):
        Z_sitk = sitk.GetImageFromArray(np.squeeze(Z[i]))
        Z_sitk.CopyInformation(data.vols_sitk[input_modalities[0]][vol_num])
        sitk.WriteImage(Z_sitk, os.path.join(output_folder,data.patient_names[vol_num]+'_'+str(i)+'.nii.gz'))
