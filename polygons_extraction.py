#!/usr/bin/env python
# -*- coding: utf-8 -*-

__project__ = 'Agricultural Irrigation Ponds Detection'
__file__ = './polygons_extraction.py'
__copyright__ = 'Copyright 2020, INIRENA-UMSNH'
__license__ = 'GPL'
__version__ = '1.0'
__date__ = 'March, 2020'
__maintainer__ = 'Jose Trinidad Soto Gonzalez'


'''

./polygons_extraction.py

First, extracts the polygons from predicted segmentation masks (Run first predict.py)
and then exports the polygons to a ESRI file (.shp)

Required Third-party libraries:
    
    * pandas
    * numpy

Required costum modules
    
    * ./functions/utils
    * ./functions/load_data
    * ./functions/georeferencing_polygons

'''

# //-----------------------------------------------------------------------------------\\

# Import required modules
import os
import re
import time
import numpy as np
import pandas as pd
import multiprocessing as mp

# Import custom modules
from functions.utils import need_time
from functions.utils import files_paths
from functions.load_data import DataGenerator
from functions.georeferencing_polygons import PolygonsExtractor
from functions.georeferencing_polygons import dissolve_polygons

# //-----------------------------------------------------------------------------------\\

# //-----------------------------------------------------------------------------------\\
# //-------------------------------Input parameters -----------------------------------\\

# //-----------------------------------------------------------------------------------\\

# Predictions

# Model date creation
weights_id = '2018-11-23_20_28_57'

# Path where predictions are stored
predictions_path = './Outputs/Validation_Images/predictions/'
predictions_path = os.path.join(predictions_path, weights_id)
data_suffix = '_data.png'
prediction_suffix = '_prediction.png'

# Threshold decision between classes
thr_decision = 0.95

batch_size = 1

# //-----------------------------------------------------------------------------------\\

# Image pre-processing
im_prep_funcs = ['zero padding', 'pick channels']
im_prep_params = {'pad_size': 20, 'idxs_channel': 0}

# //-----------------------------------------------------------------------------------\\

# //-----------------------------------------------------------------------------------\\

# Image geolocalization parameters

# The PolygonExtractor class needs the referece image center in (long, lat) decimal format. 
# But it also needs a metric coordinate referece system to reproject the reference center. 
# See geopandas documentation:
# http://geopandas.org/
crs_rep = {'init': 'epsg:6372'}

# Zoom image; see google maps documentation:
# https://developers.google.com/maps/documentation/maps-static/intro
zoom = 18

# //-----------------------------------------------------------------------------------\\

# CSV file where reference coordinates are saved for each image
images_info = './Data/Images/Validation_Images/validation_images_info.csv'

# Columns names
center_lat = 'center_lat'
center_long = 'center_long'
image_filename = 'filename'

# //-----------------------------------------------------------------------------------\\

# How many cores to use in multiprocessing
n_cores = 4

# //-----------------------------------------------------------------------------------\\

# Output_path to save de shapefile

# Create an output directory with the model ID
# to store the results

output_path = './Outputs/Validation_Images/predicted_polygons'
if not os.path.exists(output_path):
    os.mkdir(output_path)


output_path = os.path.join(output_path, weights_id)
if not os.path.exists(output_path):
    os.mkdir(output_path)
output_path = os.path.join(output_path, 'polygons.shp')

# //-----------------------------------------------------------------------------------\\

# //-----------------------------------------------------------------------------------\\
# //------------------------------ Local Functions ------------------------------------\\

# //-----------------------------------------------------------------------------------\\

def get_reference_coordinate(im_path):
    '''Helper function to retrieve the image center from csvfile'''
    global images_info
    image_name = os.path.split(im_path.replace(prediction_suffix, data_suffix))[1]
    mask = images_info.filename.apply(lambda x: re.findall(image_name, x))
    mask = mask.apply(lambda x: len(x) > 0)
    center = images_info[[center_long,center_lat]][mask].values[0]
    return center
        
# //-----------------------------------------------------------------------------------\\

def extract_polygons(idx):
    '''Pipeline to extract the polygons in one image'''
    global images

    # Get the image
    im = images[idx][0]
    im_path = images.batch_ims_paths[0]

    if idx > 0 and not idx % 100:
        print(f'Current idx : {idx}')

    # Get the reference center
    center = get_reference_coordinate(im_path)
    input_parameters = {'im' : im,
                        'center' : center,
                        'crs_rep' : crs_rep,
                        'zoom' : zoom,
                        'thr_decision' : thr_decision}
    
    # Fit the polygon extractor
    p = PolygonsExtractor(**input_parameters)

    if idx == 0:
        print('\nGeographic reference information:')
        print(f'Pixel resolution X : {p.resolution_x:.6f}')
        print(f'Pixel resolution Y : {p.resolution_y:.6f}')
        print(f'Zoom image : {p.zoom}')
        print(f'CRS : {p.crs}')
        print(f'Threshold decision : {p.thr_decision}\n')

    # Get the polygons
    polygons = list(p.get_polygons().values())

    return polygons
            
# //-----------------------------------------------------------------------------------\\

# //-----------------------------------------------------------------------------------\\
# //-----------------------------------  Run  -----------------------------------------\\

if __name__ == '__main__':

    init = time.time()

    # //-------------------------------------------------------------------------------\\

    # Load the images
    predictions_paths = files_paths(predictions_path,
                                    nested_carpets=True,
                                    exts=prediction_suffix)

    input_parameters = {'ims_paths' : predictions_paths,
                        'batch_size' : batch_size,
                        'im_prep_funcs' : im_prep_funcs,
                        'im_prep_params' : im_prep_params}

    images = DataGenerator(**input_parameters)
    
    print(f'\nExtracting polygons from predicted segmentation masks:')
    print(f'Model date creation: {weights_id}\n')
    print('Dataset information:')
    print(images)

    # //-------------------------------------------------------------------------------\\

    # Load the reference coordinates
    images_info = pd.read_csv(images_info)

    # //-------------------------------------------------------------------------------\\

    # Extract polygons in parallel processing
    pool = mp.Pool(processes=n_cores)
    polygons = pool.map(extract_polygons, range(len(images)))
    pool.close()

    # Remove empty lists
    polygons = np.concatenate(polygons)

    # //-------------------------------------------------------------------------------\\

    # Dissolve and export the polygons to an ESRI file
    polygons = dissolve_polygons(polygons)
    polygons = polygons.to_crs(crs_rep)
    polygons.to_file(driver='ESRI Shapefile', filename=output_path)

    print(f'\nSaved file at: {output_path}')

    endt = time.time()
    need_time(init, endt)

    # //-------------------------------------------------------------------------------\\    

# //-----------------------------------------------------------------------------------\\
# //--------------------------------------------------------------------------\\
# //-----------------------------------------------------------------\\
# //-----------------------------------------------------\\
# //-----------------------------------------\\
# //-----------------------------\\
# //----------------\\
# //------\\
# END