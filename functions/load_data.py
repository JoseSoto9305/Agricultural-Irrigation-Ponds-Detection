#!/usr/bin/env python
# -*- coding: utf-8 -*-

__project__ = 'Agricultural Irrigation Ponds Detection'
__file__ = './functions/load_data.py'
__copyright__ = 'Copyright 2020, INIRENA-UMSNH'
__license__ = 'GPL'
__version__ = '1.0'
__date__ = 'March, 2020'
__maintainer__ = 'Jose Trinidad Soto Gonzalez'


'''

./functions/load_data.py


Generates batches of data.


Defined classes:

    * DataGenerator


Required third-party libraries:

    * numpy
    * skimage
    * keras


Required custom modules:

    * ./functions/image.py

'''

# //-----------------------------------------------------------------------------------\\

# Import dependencies
import sys
import keras
import numpy as np

from numbers import Number

from skimage import img_as_ubyte

# Import custom modules
sys.path.append('./functions')
from image import BaseImage
from image import ImagePreprocessing
from image import DataAugmentation

# //-----------------------------------------------------------------------------------\\
            
class DataGenerator(BaseImage, keras.utils.Sequence):

    '''
    Class for generate batches of data.
    See ./training.py, ./predict.py or ./evaluation.py for a further implementation.
    '''

    def __init__(self, 
                 ims_paths,
                 labels_paths=None,
                 n_classes=None,
                 batch_size=1,
                 steps_per_epoch=None,
                 shuffle=False,
                 class_weights={},
                 im_prep_funcs=['scaling'],
                 im_prep_params={},
                 labels_prep_funcs=['pick channels'],
                 labels_prep_params={'idxs_channel': 0},
                 aug_funcs=[],
                 aug_params={}):

        # Set image pre-processing objects
        self.im_processor = ImagePreprocessing(im_prep_funcs, im_prep_params)
        if labels_prep_funcs:
            self.labels_processor = ImagePreprocessing(labels_prep_funcs,
                                                       labels_prep_params)
        else:
            self.labels_processor = None

        # Set Data Augmentation object
        if aug_funcs:
            self.augmentor = DataAugmentation(aug_funcs, aug_params)
        else:
            self.augmentor = None

        # Data information
        self.ims_paths = ims_paths
        self.n_files = len(ims_paths)
        im = self._read_image(self.ims_paths[0])
        im = self.im_processor(im)
        self.data_dtype = im.dtype
        self.h, self.w, self.channels = self._image_dim(im)

        # Labels information
        self.labels_paths = labels_paths
        self.n_classes = n_classes
        self.class_weights = class_weights

        # Batch parameters
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.indexes = np.arange(self.n_files, dtype=int)
        if shuffle:
            np.random.shuffle(self.indexes)

        # Information to display in the __repr__ and __str__ methods
        self.show_infos = ['Number of samples', 'Image height', 'Image width', 
                           'Number of channels','Number of classes']
        self.show_values = [self.n_files, self.h, self.w, self.channels, self.n_classes]

    
    def __len__(self):
        '''Denotes the number of steps per epoch'''
        if self.steps_per_epoch is None:
            return int(np.floor(self.n_files / self.batch_size))
        else:
            return self.steps_per_epoch


    def __getitem__(self, index):
        '''Retrives a batch of data'''
        if self.batch_size > 1:
            indexes = self.indexes[index*self.batch_size:
                                   (index+1)*self.batch_size]
        else:
            indexes = [index]
        return self._get_batch(indexes)



    def _labels2hot_vector(self, labels):
        '''
        Reshape labels (self.h, self.w) 
        to one-hot vectors (self.h, self.w, self.n_classes).

        If self.class_weights is not empty:
            reshape labels to (self.h * self.w, self._n_classes).

        ** Only works if n_classes == 2

        '''

        if not labels.dtype == np.uint8:
            labels = img_as_ubyte(labels)
        # Change if labels have small values
        labels[labels < 100] = 0
        labels[labels >= 100] = 255

        one_hot_vectors = np.zeros((self.h, self.w, self.n_classes), 
                                    dtype=np.float32)
        one_hot_vectors[..., 1] = labels
        one_hot_vectors[..., 0] = ~labels
        one_hot_vectors[one_hot_vectors < 100] = 0
        one_hot_vectors[one_hot_vectors >= 100] = 1

        if self.class_weights:
            one_hot_vectors = one_hot_vectors.reshape(self.h * self.w, self.n_classes)

        return one_hot_vectors


    def _sample_weights(self, sample_labels):
        '''
        Retrieves sample weights.

        ** Only works if n_classes == 2
        '''

        weights = np.zeros(self.h*self.w, dtype=np.float32)
        if self.class_weights:
            weights[np.ravel(sample_labels[...,0] == 1)] = self.class_weights[0]
            weights[np.ravel(sample_labels[...,1] == 1)] = self.class_weights[1]
        return weights


    def _get_batch(self, indexes):
        '''
        Get a batch of Data
        '''

        # Set empty vectors to store information
        self.batch_ims_paths = []
        if self.channels == 1:
            X = np.empty((self.batch_size, self.h, self.w), dtype=self.data_dtype)
        else:
            X = np.empty((self.batch_size, self.h, self.w, self.channels), dtype=self.data_dtype)

        if self.labels_paths is not None:
            if self.class_weights:
                y = np.empty((self.batch_size, self.h*self.w, self.n_classes), dtype=np.float32)
                sample_weights = np.zeros((self.batch_size, self.h*self.w), dtype=np.float32)
            else:
                y = np.empty((self.batch_size, self.h, self.w, self.n_classes), dtype=np.float32)

        
        for i, idx in enumerate(indexes):
            # Fit data
            data = self._read_image(self.ims_paths[idx])
            self.batch_ims_paths.append(self.ims_paths[idx])
            data = self.im_processor(data)

            # Fit labels
            if self.labels_paths is not None:
                labels = self._read_image(self.labels_paths[idx])
                if self.labels_processor is not None:
                    labels = self.labels_processor(labels)

            # Augment data
            if self.augmentor is not None:
                if self.labels_paths is not None:
                    data, labels = self.augmentor(data, labels)
                else:
                    data = self.augmentor(data)

            # Store in empty vectors
            X[i] = data
            if self.labels_paths is not None:
                labels = self._labels2hot_vector(labels)
                y[i] = labels
            if self.class_weights:
                sample_weights[i] = self._sample_weights(labels)

        # Retrive batch
        if self.labels_paths is None:
            return X            
        else:
            if not self.class_weights:
                return X, y
            else:
                return X, y, sample_weights


# //-----------------------------------------------------------------------------------\\

# //-----------------------------------------------------------------------------------\\
# //--------------------------------------------------------------------------\\
# //-----------------------------------------------------------------\\
# //-----------------------------------------------------\\
# //-----------------------------------------\\
# //-----------------------------\\
# //----------------\\
# //------\\
# END