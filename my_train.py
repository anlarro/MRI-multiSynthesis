import numpy as np
import matplotlib.pyplot as plt

from model import Multimodel
from my_loader import Data

from keras.callbacks import EarlyStopping

input_modalities = ['T1','T2','T2-FLAIR']
output_modalities = ['T1','T2','T2-FLAIR']
folder = '/mnt/D8D413E4D413C422/I3M/Imagenes/Oasis/data-reduced'
data = Data(folder+'/Training',folder+'/Validation')
dummy_volume = data.readCase(folder+'/Training', 0, input_modalities) #read one volume to obtain self.vol_shape

#build model
weights = {m:1.0 for m in output_modalities}
weights['concat']=1.0
m = Multimodel(input_modalities, output_modalities, weights, 16, 1, True, 'max', True, True, data.vol_shape)
m.build()

# Training params
NUM_EPOCHS = 1
BATCH_SIZE = 8

num_train_images = data.countImages(input_modalities, mode = 'train')
num_val_images = data.countImages(input_modalities, mode = 'valid')
# Configure early stopping
es = EarlyStopping(monitor='val_loss', min_delta=0.01, mode='min', patience=10)

# Initialize both the training and testing image generators
trainGen = data.generate_batches(input_modalities, output_modalities, m.num_emb, batch_size = BATCH_SIZE, mode = 'train')

valGen = data.generate_batches(input_modalities, output_modalities, m.num_emb, batch_size = BATCH_SIZE, mode = 'valid')

# train the network
print("Fitting model with generator...")
H = m.model.fit_generator(
	trainGen,
	samples_per_epoch=num_train_images,
	validation_data=valGen,
	nb_val_samples=num_val_images,
	nb_epoch=NUM_EPOCHS,
    callbacks = [es],
    verbose = 2)
m.model.save_weights('my_weights.h5')

# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig("training_plot.png")