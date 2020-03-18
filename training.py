#!/usr/bin/env python
# -*- coding: utf-8 -*-

__project__ = 'Agricultural Irrigation Ponds Detection'
__file__ = './training.py'
__copyright__ = 'Copyright 2020, INIRENA-UMSNH'
__license__ = 'GPL'
__version__ = '1.0'
__date__ = 'March, 2020'
__maintainer__ = 'Jose Trinidad Soto Gonzalez'


'''

./training.py

Trains a Fully Convolutional Network (FCN) with ResNet-50 as feature extractor.
The neural network is trained on a labeled binary dataset of agricultural irrigation ponds.
We used Google Map Static API as a source of high-resolution satellite imagery:
https://developers.google.com/maps/documentation/maps-static/intro


Required Third-party libraries:
    
    * keras
    * numpy

Required custum modules
    
    * ./functions/load_data
    * ./functions/utils
    * ./neural_network/resnet
    * ./neural_network/metrics

'''

# //-----------------------------------------------------------------------------------\\

# Required libraries
import os
import time
import numpy as np

from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler

# Custom modules
from functions.utils import files_paths
from functions.utils import need_time
from functions.utils import save_pickle
from functions.load_data import DataGenerator

from neural_network import resnet
from neural_network.metrics import dice_loss
from neural_network.metrics import dice_coeff
from neural_network.metrics import crossentropy_dice_loss

# //-----------------------------------------------------------------------------------\\


# //-----------------------------------------------------------------------------------\\
# //-------------------------------Input parameters -----------------------------------\\

# //-----------------------------------------------------------------------------------\\

seed = 1234
np.random.seed(seed)

# Image directory

# Train and validation paths
train_images_path = './Data/Images/Train_Images'
validation_images_path = './Data/Images/Validation_Images'

data_suffix = '_data.png'
labels_suffix = '_labels.png'

# //-----------------------------------------------------------------------------------\\

# Image pre-processing

im_prep_funcs = ['crop', 'scaling']
im_prep_params = {'crop_size': 20}


labels_prep_funcs = ['crop', 'pick channels']
labels_prep_params = {'crop_size': 20, 'idxs_channel': 0}

aug_funcs = ['flip', 'rotate']
aug_params = {'p' : 0.5}

# //-----------------------------------------------------------------------------------\\

# Training parameters

n_classes = 2
class_weights = {0: 0.5, 1: 1.5}

shuffle = True
epochs = 1
steps_per_epoch = 1
train_batch_size = 1
validation_batch_size = 1
validation_steps = 1

learning_rate = 1.0

# `RMSprop`, `Adam` 
optimizer = 'RMSprop'

# `crossentropy dice loss`, `dice loss`
loss = 'crossentropy dice loss'

# `dice_coeff`, `[]`
accuracy = '[]'

# Steps without any improve before stop
early_stop_steps = 4

# Load pretrained ResNet with ImageNet
use_pretreining_imagent = True

# Freeze pretrained layers
freeze = False

# //-----------------------------------------------------------------------------------\\

# Outputs, create a carpet with the current time
t = time.localtime()
base_name = '{}-{}-{}_{}_{}_{}'.format(t.tm_year,
                                       t.tm_mon,
                                       t.tm_mday,
                                       t.tm_hour,
                                       t.tm_min,
                                       t.tm_sec)

base_name = f'./neural_network/Model/{base_name}'
if not os.path.exists(base_name):
    os.mkdir(base_name)
    os.mkdir(os.path.join(base_name, 'log'))

path_save_input_params = os.path.join(base_name, 'input_params.pkl')
path_save_weights = os.path.join(base_name, 'weights.h5')
path_tensorboard_log = os.path.join(base_name, 'log', 'log_out')

# //-----------------------------------------------------------------------------------\\

# //-----------------------------------------------------------------------------------\\
# //-----------------------------------  Run  -----------------------------------------\\

if __name__ == '__main__':
    
    init = time.time()

    # //-------------------------------------------------------------------------------\\

    # Set the train paths
    train_ims_paths = files_paths(train_images_path, 
                                  nested_carpets=True, exts=data_suffix)
    train_labels_paths = [i.replace(data_suffix, 
                                    labels_suffix) for i in train_ims_paths]

    # Set the validation paths
    validation_ims_paths = files_paths(validation_images_path, 
                                       nested_carpets=True, exts=data_suffix)
    validation_labels_paths = [i.replace(data_suffix, 
                                         labels_suffix) for i in validation_ims_paths]
    
    # //-------------------------------------------------------------------------------\\

    input_train_parameters = {'ims_paths' : train_ims_paths,
                              'labels_paths' : train_labels_paths,
                              'class_weights' : class_weights,
                              'batch_size' : train_batch_size,
                              'steps_per_epoch' : steps_per_epoch,
                              'shuffle' : shuffle,
                              'n_classes' : n_classes,
                              'im_prep_funcs' : im_prep_funcs,
                              'im_prep_params': im_prep_params,
                              'labels_prep_funcs' : labels_prep_funcs,
                              'labels_prep_params' : labels_prep_params,
                              'aug_funcs' : aug_funcs,
                              'aug_params' : aug_params}

    input_validation_parameters = {'ims_paths' : validation_ims_paths,
                                   'labels_paths' : validation_labels_paths,
                                   'class_weights' : class_weights,
                                   'batch_size' : validation_batch_size,
                                   'steps_per_epoch' : validation_steps,
                                   'shuffle' : shuffle,
                                   'n_classes' : n_classes,
                                   'im_prep_funcs' : im_prep_funcs,
                                   'im_prep_params': im_prep_params,
                                   'labels_prep_funcs' : labels_prep_funcs,
                                   'labels_prep_params' : labels_prep_params}

    # //-------------------------------------------------------------------------------\\

    # Fit the train and validation data generators
    train_generator = DataGenerator(**input_train_parameters)
    validation_generator = DataGenerator(**input_validation_parameters)
    
    print('\nTrain set information:')
    print(train_generator)
    print('\nValidation set information:')
    print(validation_generator)

    # //-------------------------------------------------------------------------------\\

    # Create the model

    input_shape = [train_generator.h, 
                   train_generator.w, 
                   train_generator.channels]

    model = resnet.ResNet50_FCN(input_shape, n_classes, 
                                use_pretreining_imagent, freeze)
    # Optimizer
    if optimizer == 'RMSprop':
        optimizer_function = RMSprop(lr=learning_rate) 
    elif optimizer == 'Adam':
        optimizer_function = Adam(lr=learning_rate)

    # Loss function
    if loss == 'crossentropy dice loss':
        loss_function = crossentropy_dice_loss 
    elif loss == 'dice loss':
        loss_function = dice_loss

    # Accuracy
    if accuracy == 'dice_coeff':
        accuracy_function = [dice_coeff]

    else:
        accuracy_function = []
    
    # Compile the model
    if class_weights is not None:    
        model.compile(optimizer=optimizer_function, 
                      loss=loss_function, 
                      metrics=accuracy_function,
                      sample_weight_mode='temporal')
    else:
        model.compile(optimizer=optimizer_function, 
                      loss=loss_function, 
                      metrics=accuracy_function)

    # Training callbacks
    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=early_stop_steps,
                               verbose=1,
                               min_delta=1e-4),                 
    
                 LearningRateScheduler(resnet.lr_scheduler_function),

                 ModelCheckpoint(monitor='val_loss',
                                 filepath=path_save_weights,
                                 save_best_only=True,
                                 save_weights_only=True),
                 
                 TensorBoard(log_dir=path_tensorboard_log)]

    # //-------------------------------------------------------------------------------\\
    
    # Save the input parameters
    params = {'seed' : seed,

              # Image directory
              'train_images_path' : train_images_path,
              'validation_images_path' : validation_images_path,
              'data_suffix' : data_suffix,
              'labels_suffix' : labels_suffix,

              # Image pre-processing
              'im_prep_params': im_prep_params,
              'im_prep_funcs' : im_prep_funcs,
              'labels_prep_params' : labels_prep_params,
              'labels_prep_funcs' : labels_prep_funcs,
              'aug_params' : aug_params,
              'aug_funcs' : aug_funcs,

              # Training parameters
              'n_train_files' : train_generator.n_files,
              'n_validation_files' : validation_generator.n_files,
              'input_shape' : input_shape,
              'n_classes' : n_classes,

              'epochs' : epochs,
              'shuffle' : shuffle,
              'steps_per_epoch' : steps_per_epoch,
              'train_batch_size' : train_batch_size,

              'validation_batch_size' : validation_batch_size,
              'validation_steps' : validation_steps,

              'learning_rate' : learning_rate,
              'optimizer' : optimizer,
              'loss' : loss,
              'accuracy' : accuracy,
              'class_weights' : class_weights,

              'early_stop_steps' : early_stop_steps,
              'use_pretreining_imagent' : use_pretreining_imagent,
              'freeze' : freeze,

              # Outputs
              'output_identifier' : base_name,
              'weights_output' : path_save_weights
              }

    save_pickle(params, path_save_input_params)
    
    # //-------------------------------------------------------------------------------\\

    # Fit the model with generators
    hist = model.fit_generator(generator=train_generator,
                               validation_data=validation_generator,
                               epochs=epochs,                               
                               validation_steps=validation_steps,
                               callbacks = callbacks)
    
    # //-------------------------------------------------------------------------------\\

    print(f'Model saved at: {path_save_weights}')
    
    endt = time.time()
    need_time(init, endt)

# //-----------------------------------------------------------------------------------\\
# //--------------------------------------------------------------------------\\
# //-----------------------------------------------------------------\\
# //-----------------------------------------------------\\
# //-----------------------------------------\\
# //-----------------------------\\
# //----------------\\
# //------\\
# END