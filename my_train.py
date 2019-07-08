import math
import numpy as np

from model import Multimodel
from my_loader import Data

from keras.callbacks import EarlyStopping

input_modalities = ['T1','T2']
output_modalities = ['PD']
folder = '/mnt/D8D413E4D413C422/I3M/Imagenes/IXI/data-reduced'

data = Data(folder, input_modalities + output_modalities)
data.load()

#build model
weights = {m:1.0 for m in output_modalities}
weights['concat']=1.0
m = Multimodel(input_modalities, output_modalities, weights, 16, 1, False, 'max', True, True)
m.build()

#Prepare data
trainFrac = math.floor(data.num_vols*0.7)
valFrac = math.floor(data.num_vols*0.1)

ids_train = range(trainFrac)
ids_val = range(trainFrac,trainFrac+valFrac)

train_in = [data.get(mod, ids_train) for mod in input_modalities]
valid_in = [data.get(mod, ids_val) for mod in input_modalities]

# there's 1 output per embedding plus 1 output for the total variance embedding
train_out = [data.get(mod, ids_train) for mod in output_modalities for i in range(m.num_emb)]
valid_out = [data.get(mod, ids_val) for mod in output_modalities for i in range(m.num_emb)]

train_shape = (train_out[0].shape[0], 1, train_out[0].shape[2], train_out[0].shape[3])
valid_shape = (valid_out[0].shape[0], 1, valid_out[0].shape[2], valid_out[0].shape[3])
if len(input_modalities) > 1:
    train_out += [np.zeros(shape=train_shape) for i in range(2)]
    valid_out += [np.zeros(shape=valid_shape) for i in range(2)]

#train
es = EarlyStopping(monitor='val_loss', min_delta=0.01, mode='min', patience=10)

print('Fitting model...')
m.model.fit(train_in, train_out, validation_data=(valid_in, valid_out), nb_epoch=1, batch_size=32,callbacks=[es])
m.model.save_weights('my_weights.h5')