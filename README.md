# Multimodal Brain Synthesis

This is my own adaptation of the [multimodal_brain_synthesis] project. For details see
the papers [Multimodal MR Synthesis via Modality-Invariant Latent Representation] and [Robust Multi-Modal MR Image Synthesis].

The main files in this project are:

* model.py: contains the neural network implementation
* my_loader.py: loads the input data into a Data object.
* my_train.py: loads the specified data, trains and saves the model.
* my_test.py: loads the specified data, predicts and saves predicted volumes.

The code is written in Keras and expects image_data_format to be set to channels_first (theano backend).

[Multimodal MR Synthesis via Modality-Invariant Latent Representation]: http://ieeexplore.ieee.org/document/8071026/
[Robust Multi-Modal MR Image Synthesis]: https://link.springer.com/chapter/10.1007/978-3-319-66179-7_40
[multimodal_brain_synthesis]: https://github.com/agis85/multimodal_brain_synthesis



