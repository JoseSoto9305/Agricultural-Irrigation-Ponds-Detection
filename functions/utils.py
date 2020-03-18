#!/usr/bin/env python
# -*- coding: utf-8 -*-

__project__ = 'Agricultural Irrigation Ponds Detection'
__file__ = './functions/utils.py'
__copyright__ = 'Copyright 2020, INIRENA-UMSNH'
__license__ = 'GPL'
__version__ = '1.0'
__date__ = 'March, 2020'
__maintainer__ = 'Jose Trinidad Soto Gonzalez'


'''

./functions/utils.py


Miscellaneous functions.


Defined functions:

    * files_paths
    * load_pickle
    * need_time
    * save_pickle

Required third-party libraries:

    * numpy

'''

# //-----------------------------------------------------------------------------------\\

# Import dependencies

import os
import pickle
import numpy as np

# //-----------------------------------------------------------------------------------\\

def files_paths(path, nested_carpets=False, exts=None):
    '''
    Returns a list with file paths in a given directory.

    By default looks for image files.

    If nested_carpets:
        Apply os.walk(input_path)

    '''
    
    # Image file extentions
    if exts is None:
        exts = ['.jpg','.JPG','.png','.PNG',
               '.tif','.TIF','.tiff','.TIFF']
    else:
        if not isinstance(exts, list):
            exts = [exts]
        
    if nested_carpets:
        # Look for files in nested carpets
        files = []
        for root, dirs, filess in os.walk(path):
            files.extend([os.path.join(root, f) for f in filess 
                          if any(f.endswith(j) for j in exts)])
    
    else:
        # Look for files in the input directory        
        # Omit nested carpets
        files = [os.path.join(path,i) for i in os.listdir(path) 
                if os.path.isfile(os.path.join(path,i))]
        files = [i for i in files if any(i.endswith(j) for j in exts)]
    
    return sorted(files)

# //-----------------------------------------------------------------------------------\\

def load_pickle(path, is_list=False):
    '''
    Loads and reads a pickle file.
    '''

    if not is_list:
        f = open(path, 'rb')
        data = pickle.load(f)
        return data
    
    else:
        f = open(path, 'rb')
        data = []

        while True:
            try:
                tmp = pickle.load(f)
                data.append(tmp)
            except EOFError:
                break
        
        return np.array(data)

# //-----------------------------------------------------------------------------------\\

def need_time(init, endt):
    '''
    Prints on shell the required time for an 
    action in minutes.
    '''

    t = (endt - init) / 60
    print(f'\nRequired time: {t:.4f} minutes \n')


# //-----------------------------------------------------------------------------------\\

def save_pickle(obj, path, append_mode=False):
    '''
    Saves an object in a pickle file.
    '''

    if append_mode:
        with open(path, 'a') as f:
            pickle.dump(obj, f)

    else:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

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