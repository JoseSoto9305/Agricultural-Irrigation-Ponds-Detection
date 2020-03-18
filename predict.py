#!/usr/bin/env python
# -*- coding: utf-8 -*-

__project__ = 'Agricultural Irrigation Ponds Detection'
__file__ = './predict.py'
__copyright__ = 'Copyright 2020, INIRENA-UMSNH'
__license__ = 'GPL'
__version__ = '1.0'
__date__ = 'March, 2020'
__maintainer__ = 'Jose Trinidad Soto Gonzalez'


'''

./predict.py

This script, uses the trained model from ./training.py 
to predict the segmentation of agricultural irrigation ponds in google map satellite imagery.

Required Third-party modules:
    
    * skimage

Required custom modules:
    
    * ./functions/load_data
    * ./functions/utils
    * ./functions/image
    * ./neural_network/resnet

'''

# //-----------------------------------------------------------------------------------\\

# Required modules
import os
import time
import warnings
from skimage.io import imsave

# Custom modules

from neural_network.resnet import restore_model

from functions.utils import need_time
from functions.utils import files_paths
from functions.load_data import DataGenerator
from functions.image import display_image_and_labels
from functions.image import display_image_labels_and_prediction

# //-----------------------------------------------------------------------------------\\

# //-----------------------------------------------------------------------------------\\
# //-------------------------------Input parameters -----------------------------------\\

# //-----------------------------------------------------------------------------------\\

# Path where images are stored
validation_images_path = './Data/Images/Validation_Images'
data_suffix = '_data.png'

# if ground truth labels are available; else None
labels_suffix = '_labels.png'

# If dataset_n_images % batch_size > 0:
#    The remainder won't pass
batch_size = 1

n_classes = 2
shuffle = False

# //-----------------------------------------------------------------------------------\\

# Image pre-processing

im_prep_funcs = ['crop', 'scaling']
im_prep_params = {'crop_size':20}

# if ground truth labels are available; else empty
labels_prep_funcs = ['pick channels', 'crop']
labels_prep_params = {'idxs_channel': 0, 'crop_size':20}

# //-----------------------------------------------------------------------------------\\

# Restoring the model

# Model name
weights_id = '2018-11-23_20_28_57'

weights_path = f'./neural_network/Model/{weights_id}/weights.h5'

# //-----------------------------------------------------------------------------------\\

# Saving predictions
save_predictions = True
prediction_suffix = '_prediction.png'

# Creating an output directory with the model ID
# to store the predictions
output_path = './Outputs/Validation_Images/predictions'
if not os.path.exists(output_path):
    os.mkdir(output_path)

output_path = os.path.join(output_path, weights_id)
if not os.path.exists(output_path):
    os.mkdir(output_path)

# //-----------------------------------------------------------------------------------\\

show_predictions = True

# //-----------------------------------------------------------------------------------\\

# //-----------------------------------------------------------------------------------\\
# //-----------------------------------  Run  -----------------------------------------\\

if __name__ == '__main__':
    
    init = time.time()

    # //-------------------------------------------------------------------------------\\

    # Load the images paths
    validation_ims_paths = files_paths(validation_images_path, 
                                       nested_carpets=True, exts=data_suffix)
    # Look for labels paths
    if labels_suffix is not None:
        labels_paths = [i.replace(data_suffix, 
                                  labels_suffix) for i in validation_ims_paths]        
        labels_paths = [i for i in labels_paths if os.path.exists(i)]
        
        if not labels_paths:
            labels_paths = None
    else:
        labels_paths = None

    # //-------------------------------------------------------------------------------\\

    input_parameters = {'ims_paths': validation_ims_paths,
                        'labels_paths' : labels_paths,
                        'batch_size' : batch_size,
                        'n_classes' : n_classes,
                        'shuffle' : shuffle,                        
                        'im_prep_funcs' : im_prep_funcs,
                        'im_prep_params': im_prep_params,
                        'labels_prep_funcs' : labels_prep_funcs,
                        'labels_prep_params' : labels_prep_params}

    # //-------------------------------------------------------------------------------\\

    # Fit the data generator
    validation_images = DataGenerator(**input_parameters)
    print(validation_images)

    # //-------------------------------------------------------------------------------\\

    # Restore the trained model
    input_shape = (validation_images.h, 
                   validation_images.w, 
                   validation_images.channels)
    model = restore_model(input_shape, n_classes, weights_path)
    
    # //-------------------------------------------------------------------------------\\

    for i in range(len(validation_images)):

        print(f'Current batch: {i}')
        if labels_paths is not None:
        	X, y = validation_images[i]
        else:
        	X = validation_images[i]

        # Make prediction
        y_pred = model.predict(X).reshape(validation_images.batch_size, 
                                          validation_images.h, 
                                          validation_images.w, 
                                          validation_images.n_classes)
    
        for j in range(y_pred.shape[0]):

            if save_predictions:
                # Create an output directory for each image
                direct, im_name = os.path.split(validation_images.batch_ims_paths[j])
                p = os.path.join(output_path, os.path.basename(direct))            
                if not os.path.exists(p):
                    os.mkdir(p)                        
                output_name = os.path.join(p, im_name.replace(data_suffix,
                                                              prediction_suffix))
                # Save
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    imsave(output_name, y_pred[j,...,1])
       
            if show_predictions:
                if labels_paths is None:                    
                    display_image_and_labels(X[j,...], 
                                             y_pred[j,...,1], colormap='jet')
                else:
                    display_image_labels_and_prediction(X[j,...], 
                                                        y[j,...,1], 
                                                        y_pred[j,...,1])
    
    if save_predictions:
        print(f'\nPredictions saved at: {output_path}')

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