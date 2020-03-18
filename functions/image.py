#!/usr/bin/env python
# -*- coding: utf-8 -*-

__project__ = 'Agricultural Irrigation Ponds Detection'
__file__ = './functions/image.py'
__copyright__ = 'Copyright 2020, INIRENA-UMSNH'
__license__ = 'GPL'
__version__ = '1.0'
__date__ = 'March, 2020'
__maintainer__ = 'Jose Trinidad Soto Gonzalez'


'''

./functions/image.py


Functions to apply image pre-processing


Defined classes:
    
    * BaseImage
    * BaseProcessing
    * DataAugmentation
    * ImagePreprocessing

Defined functions:

    * cropping
    * display_image
    * display_image_and_labels
    * display_image_labels_and_prediction
    * display_overlap_images
    * im_rescaling
    * pick_channels    
    * read_image
    * reshape
    * zero_padding

Requiered Third-party libraries

    * matplotlib
    * numpy
    * skimage

'''

# //-----------------------------------------------------------------------------------\\

# Import dependencies

import os
import numpy as np
import matplotlib.pyplot as plt

from numpy import random

from skimage.io import imread
from skimage import img_as_float
from skimage import img_as_ubyte
from skimage.color import rgb2hsv
from skimage.color import hsv2rgb
from skimage.filters import gaussian
from skimage.transform import rotate
from skimage.transform import resize
from skimage.util import random_noise
from skimage.exposure import adjust_gamma

# //-----------------------------------------------------------------------------------\\

def cropping(im, crop_size=20):
    '''
    Crop image, apply the size 
    along each image border
    '''
    return im[crop_size:im.shape[0]-crop_size, 
              crop_size:im.shape[1]-crop_size]

# //-----------------------------------------------------------------------------------\\

def display_image(im, colormap=None):
    '''
    Displays the input image on the screen.
    '''
    fig = plt.figure(figsize=(12,6))
    plt.imshow(im, cmap=colormap)
    plt.title('Image')
    plt.xticks([])
    plt.yticks([])
    plt.show()

# //-----------------------------------------------------------------------------------\\
    
def display_image_and_labels(im, labels, colormap='gray'):
    '''
    Displays input image and labels on the screen.
    '''

    fig, ax = plt.subplots(1,2, sharey=True, figsize=(12,6))
    ax[0].imshow(im)
    ax[0].set_title('Image')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    if len(labels.shape)>2:
        labels = labels[...,0]
    cb = ax[1].imshow(labels, cmap=colormap)
    ax[1].set_title('Labels')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    cax = plt.axes([0.92, 0.2, 0.01, 0.6])
    plt.colorbar(cb, cax=cax)
    plt.show()

# //-----------------------------------------------------------------------------------\\

def display_image_labels_and_prediction(im, labels, pred):
    '''
    Displays input image, labels and prediction on the screen.
    '''

    fig, ax = plt.subplots(1,3, sharey=True, figsize=(12,6))
    ax[0].imshow(im)
    ax[0].set_title('Image')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    if len(labels.shape)>2:
        labels = labels[...,0]
    ax[1].imshow(labels, cmap='gray')
    ax[1].set_title('Ground Truth')
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    if len(pred.shape) > 2:
            pred = pred[...,0]
    pred = img_as_float(pred)
    cb = ax[2].imshow(pred, cmap='jet')
    ax[2].set_title('Prediction')
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    cax = plt.axes([0.92, 0.2, 0.01, 0.6])
    plt.colorbar(cb, cax=cax)
    plt.show()

# //-----------------------------------------------------------------------------------\\

def display_overlap_images(im1, im2, colormap='jet'):
    '''
    Displays and blends two images.
        ** im2 must be in a one-channel format.
    '''

    fig = plt.figure(figsize=(12,6))

    if len(im2.shape)>2:
        im2 = im2[...,0]    
    im2 = img_as_float(im2)

    cb = plt.imshow(im2, cmap=colormap)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(im1, alpha=0.4)
    plt.xticks([])
    plt.yticks([])

    plt.title('Image')
    cax = plt.axes([0.75, 0.2, 0.01, 0.6])
    plt.colorbar(cb, cax=cax)
    plt.show()

# //-----------------------------------------------------------------------------------\\

def im_rescaling(im, clip_min=0., clip_max=1.):
    '''
    
    Image normalization.

                                              (clip_max - clip_min)
    Image_normalized =   (Image - Image.min) _________________________+ clip_min
                        
                                              (Image.max - Image.min)

    Reference:
    https://stackoverflow.com/questions/33610825/normalization-in-image-processing
    '''
    new_im = (im-im.min()) * ((clip_max-clip_min) / 
             (im.max() - im.min())) + clip_min
    
    return new_im

# //-----------------------------------------------------------------------------------\\

def pick_channels(im, idxs_channels=0):
    '''
    Pick channels from image.
        ** Input image must be in (h, w, n_channels) format.
    '''
    if len(im.shape) > 2:
        im = im[..., idxs_channels]
    return im    

# //-----------------------------------------------------------------------------------\\

def read_image(im_path):
    '''
    Loads into memory an image file

        ** Omit the alpha channel in RGB images
    '''
    im = imread(im_path)
    # Omit alpha channel
    if im.shape[-1] == 4:
        im = im[...,:3]
    return im

# //-----------------------------------------------------------------------------------\\

def reshape(im, new_shape=[256,256]):
    '''Resize an image to a new size'''
    return img_as_ubyte(resize(im, new_shape))

# //-----------------------------------------------------------------------------------\\

def zero_padding(im, pad_size=20):
    '''    
    Adds a zero pad for each border. 
    '''

    # New image dimentions
    new_shape = [im.shape[0] + (pad_size*2), 
                 im.shape[1] + (pad_size*2)]    
    if len(im.shape) > 2:
        new_shape.append(im.shape[-1])
    
    # Padding  
    new_im = np.zeros(new_shape, dtype=im.dtype)
    new_im[pad_size:new_shape[0]-pad_size, 
           pad_size:new_shape[1]-pad_size,] = im
    
    return new_im

# //-----------------------------------------------------------------------------------\\

class BaseImage:

    '''
    See ./functions/georeferencing_polygons.ImageGeoreferencer
    '''

    def _image_dim(self, im):
        '''Retrives image dimention (h, w, n_channels)'''
        if len(im.shape) == 2:
            self.channels = 1
        else:
            self.channels = im.shape[-1]
        self.h = im.shape[0]
        self.w = im.shape[1]
        return self.h, self.w, self.channels

    
    def _read_image(self, im_path):
        return read_image(im_path)


    def __repr__(self):
        info = f'\n{self.__class__.__name__} object \n'
        info += f'\nInformation:\n\n'
        for inf, value in zip(self.show_infos, self.show_values):
            if isinstance(value, float):
                info += f'      {inf} : {value:.4f}\n'
            else:
                info += f'      {inf} : {value}\n'                
        return info


    def __str__(self):
        return self.__repr__()
    
# //-----------------------------------------------------------------------------------\\

class BaseProcessing:

    '''
    Parent class for image processing

    See ./functions/image.ImagePreprocessing or
        ./functions/image.DataAugmentation for a further implementation
    '''

    def __init__(self, applyfuncs=[]):
        if applyfuncs == 'all':
            self._applyfuncs = self.dict_funcs.keys()
        else:
            self._applyfuncs = applyfuncs
        self._sort_applyfuncs()


    @property
    def applyfuncs(self):
        return self._applyfuncs


    @applyfuncs.setter
    def applyfuncs(self, applyfuncs=[]):
        if not isinstance(applyfuncs, list):
            applyfuncs = [applyfuncs]
        self._applyfuncs = applyfuncs
        self._sort_applyfuncs()


    def _sort_applyfuncs(self):
        keys = list(self.dict_funcs.keys())
        idxs = sorted(keys.index(i) for i in self._applyfuncs)
        self._applyfuncs = np.array(keys)[idxs].tolist()


    def __call__(self, data, labels=None):
        for func in self.applyfuncs:
            if func in ['flip', 'rotate']:
                data, labels = self.dict_funcs[func](data, labels)
            else:
                data = self.dict_funcs[func](data)
        if labels is None:
            return data
        else:
            return data, labels


    def __repr__(self):
        return f'\n{self.__class__.__name__} object:\n\n  Apply funcs:\n\n  {self._applyfuncs}\n'


    def __str__(self):
        return self.__repr__()

# //-----------------------------------------------------------------------------------\\

class ImagePreprocessing(BaseProcessing):

    '''
    Class to apply image pre-processing.
    See ./functions/load_data.DataGenerator for a further implementation.
    '''

    def __init__(self, applyfuncs=['scaling'], params={}):

        self.dict_funcs = {'pick channels': self._pick_channels,
                           'crop': self._cropping,
                           'zero padding' : self._zero_padding,
                           'reshape': self._reshape,
                           'scaling': self._normalization}

        self.params = {'clip_min' : 0.0,
                       'clip_max' : 1.0,
                       'crop_size' : 20,
                       'new_shape' : [256,256],
                       'pad_size' : 20,
                       'idxs_channels': 0}
        self.params = {**self.params, **params}

        # Initializing parent class
        BaseProcessing.__init__(self, applyfuncs)


    def _pick_channels(self, data):
        data = pick_channels(data, self.params['idxs_channels'])
        return data


    def _cropping(self, data):
        data = cropping(data, self.params['crop_size'])
        return data


    def _zero_padding(self, data):
        data = zero_padding(data, self.params['pad_size'])
        return data


    def _reshape(self, data):
        data = reshape(data, self.params['new_shape'])
        return data


    def _normalization(self, data):
        # Normalizing the image
        data = im_rescaling(data, self.params['clip_min'], 
                                  self.params['clip_max'])
        return data

# //-----------------------------------------------------------------------------------\\

class DataAugmentation(BaseProcessing):

    '''
    Class to apply Data Augmentation.
    See ./functions/load_data.DataGenerator for a further implementation.
    '''

    def __init__(self, applyfuncs=['rotate'], params={}):

        self.dict_funcs = {'coloring' : self._random_coloring,
                           'noise' : self._add_noise,
                           'intensity' : self._change_intensity,
                           'blurring' : self._blurring_image,
                           'flip' : self._flip_image,
                           'rotate' : self._rotate_image}

        self.params = {'p' : 0.5, 
                       'hue_limit' : (0, 0.1), 
                       'sat_limit' : (0, 0.3),
                       'val_limit' : (0, 0.3), 
                       'noise_max_var' : 0.04,
                       'gamma_limit' : (0.4, 3.0),
                       'sigma_limit' : (0, 2.5),
                       'r_limit' : (1, 359)}
        self.params = {**self.params, **params}

        # Initializing parent class
        BaseProcessing.__init__(self, applyfuncs)


    def _random_coloring(self, data):        
        '''
        Changes color in image adding a random hsv value
        '''

        if random.random() < self.params['p']:
            c = random.choice(['random_hsv','tinting'])
            if c == 'random_hsv':
                # Convert the image to hsv and then split each channel
                hsv = rgb2hsv(data)
                h,s,v = [hsv[...,i] for i in range(3)]
                
                # Random value for each channel
                hue = random.choice(np.linspace(self.params['hue_limit'][0], 
                                                self.params['hue_limit'][1], 
                            (self.params['hue_limit'][1]-self.params['hue_limit'][0])*1000))
                sat = random.choice(np.linspace(self.params['sat_limit'][0], 
                                                self.params['sat_limit'][1], 
                            (self.params['sat_limit'][1]-self.params['sat_limit'][0])*1000))
                val = random.choice(np.linspace(self.params['val_limit'][0], 
                                                self.params['val_limit'][1], 
                            (self.params['val_limit'][1]-self.params['val_limit'][0])*1000))

                # Add the random value at each channel
                s += sat
                h += hue
                v += val

                # Return the new image as RGB
                data = np.stack((h, s, v), axis=2)
                data = data.clip(0,1)
                data = img_as_float(hsv2rgb(data))

            elif c == 'tinting':
                hue = random.random()
                hsv = rgb2hsv(data)
                hsv[:, :, 0] = hue
                data = img_as_float(hsv2rgb(hsv))
        return data

    
    def _add_noise(self, data):
        '''
        Adds a gaussian random noise in the image.
        '''
        if random.random() < self.params['p']:
            data = random_noise(data, var=random.random()*self.params['noise_max_var'])
            data = np.clip(img_as_float(data), 0., 1.)
        return data

    
    def _change_intensity(self, data):
        '''
        Changes image intensity with gamma correction.
        '''
        if random.random() < self.params['p']:
            gamma = random.choice(np.linspace(self.params['gamma_limit'][0],
                                              self.params['gamma_limit'][1],
                        (self.params['gamma_limit'][1]-self.params['gamma_limit'][0])*100))
            data = np.clip(img_as_float(adjust_gamma(data, 
                                      gamma=gamma)), 0., 1.)
        return data

    
    def _blurring_image(self, data):
        '''
        Blurs the image with a gaussin kernel.
        '''
        if random.random() < self.params['p']:
            sigma = random.choice(np.linspace(self.params['sigma_limit'][0],
                                              self.params['sigma_limit'][1],
                        (self.params['sigma_limit'][1]-self.params['sigma_limit'][0])*100))
            data = np.clip(img_as_float(gaussian(data, 
                              sigma=sigma, multichannel=True)), 0., 1.)
        return data

    
    def _flip_image(self, data, labels=None):
        '''
        Flips the image and labels horizontally or vertically.
        '''
        if random.random() < self.params['p']:
            flip = random.choice(['h','v'])
            if flip == 'h':
                # Horizontal flip
                data = data[:, ::-1]
                if labels is not None:
                    labels = labels[:, ::-1]
            elif flip == 'v':
                # Vertical flip
                data = data[::-1, :]
                if labels is not None:
                    labels = labels[::-1, :]
        return data, labels

    
    def _rotate_image(self, data, labels=None):
        '''
        Rotates the image and labels
        '''
        if random.random() < self.params['p']:
            angle = random.randint(self.params['r_limit'][0], self.params['r_limit'][1])
            data = img_as_float(rotate(data, angle))
            if labels is not None:
                labels = img_as_float(rotate(labels, angle))
        return data, labels

# //-----------------------------------------------------------------------------------\\
# //--------------------------------------------------------------------------\\
# //-----------------------------------------------------------------\\
# //-----------------------------------------------------\\
# //-----------------------------------------\\
# //-----------------------------\\
# //----------------\\
# //------\\
# END