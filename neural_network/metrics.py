#!/usr/bin/env python
# -*- coding: utf-8 -*-

__project__ = 'Agricultural Irrigation Ponds Detection'
__file__ = './neural_network/metrics.py'
__copyright__ = 'Copyright 2020, INIRENA-UMSNH'
__license__ = 'GPL'
__version__ = '1.0'
__date__ = 'March, 2020'
__maintainer__ = 'Jose Trinidad Soto Gonzalez'


'''

./neural_network/metrics.py


Metrics


Defined functions:
    
    * crossentropy_dice_loss
    * dice_coeff
    * dice_loss

Require Third-party libraries

    * keras

'''

# //-----------------------------------------------------------------------------------\\

# Required modules
from keras import backend as K
from keras.losses import categorical_crossentropy

# //-----------------------------------------------------------------------------------\\

def dice_coeff(y_true, y_pred):
    '''
    Reference: https://stackoverflow.com/questions/31273652/how-to-calculate-dice-coefficient-for-measuring-accuracy-of-image-segmentation-i
    '''
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# //-----------------------------------------------------------------------------------\\

def dice_loss(y_true, y_pred):
    return 1-dice_coeff(y_true, y_pred)

# //-----------------------------------------------------------------------------------\\

def crossentropy_dice_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

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