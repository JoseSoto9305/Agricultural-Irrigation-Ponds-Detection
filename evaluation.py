#!/usr/bin/env python
# -*- coding: utf-8 -*-

__project__ = 'Agricultural Irrigation Ponds Detection'
__file__ = './evaluation.py'
__copyright__ = 'Copyright 2020, INIRENA-UMSNH'
__license__ = 'GPL'
__version__ = '1.0'
__date__ = 'March, 2020'
__maintainer__ = 'Jose Trinidad Soto Gonzalez'


'''

./evaluation.py

Evaluates the performances of the trained model:
First extracts the validation and prediction polygons.
Then, builds the precision-recall curve and retrives the best performance.

Our fine-tuned model detects irrigation ponds greater than 230 m² of
with a F1 score of 0.91 (Recall=0.90, Precision=0.92)

Metrics used:

    * Dice score
    * Precision
    * Recall
    * F1 Score

    **** This code takes to much time. Improves are welcome


Required Third-party libraries:
    
    * geopandas
    * numpy
    * pandas

Requiered custom modules:
    
    * ./functions/utils
    * ./functions/load_data
    * ./functions/georeferencing_polygons
    * ./functions/evaluate_functions

'''

# //-----------------------------------------------------------------------------------\\

# Required libraries
import os
import re
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import multiprocessing as mp

# Custom modules
from functions.utils import files_paths
from functions.utils import load_pickle
from functions.utils import save_pickle
from functions.utils import need_time
from functions.load_data import DataGenerator
from functions import evaluate_functions as evf
from functions import georeferencing_polygons as rp

# //-----------------------------------------------------------------------------------\\

# //-----------------------------------------------------------------------------------\\
# //-------------------------------Input parameters -----------------------------------\\

# //-----------------------------------------------------------------------------------\\

# Path where validation images are stored
validation_images_path = './Data/Images/Validation_Images'
data_suffix = '_data.png'
labels_suffix = '_labels.png'

# Model date creation
weights_id = '2018-11-23_20_28_57'

# Path where predictions are stored
predictions_path = './Outputs/Validation_Images/predictions/'
predictions_path = os.path.join(predictions_path, weights_id)
prediction_suffix = '_prediction.png'

# CSV file where reference coordinates are saved for each image
validation_images_info = './Data/Images/Validation_Images/validation_images_info.csv'

# Columns names
center_lat = 'center_lat'
center_long = 'center_long'
image_filename = 'filename'

batch_size = 1

# //-----------------------------------------------------------------------------------\\

# Image pre-processing
val_prep_funcs = ['crop', 'zero padding', 'pick channels']
val_prep_params = {'crop_size':20, 'pad_size': 20, 'idxs_channel': 0}


pred_prep_funcs = ['zero padding', 'pick channels']
pred_prep_params = {'pad_size': 20, 'idxs_channel': 0}

# //-----------------------------------------------------------------------------------\\

# Image geolocalization parameters

# The PolygonExtractor class needs the referece image center in (long, lat) decimal format. 
# But it also needs a metric coordinate referece system to reproject the reference center. 
# See geopandas documentation:
# http://geopandas.org/
crs_rep = {'init':'epsg:6372'}

# Zoom image; see google maps documentation:
# https://developers.google.com/maps/documentation/maps-static/intro
zoom = 18

# Threshold decision
thresholds = np.linspace(0.01, .99, 50)

# Threshold area in meters
thr_areas = np.array([0, 90, 160, 230])

# //-----------------------------------------------------------------------------------\\

# Cores to use in parallel process
n_cores = 4

# //-----------------------------------------------------------------------------------\\

# Outputs

# Creating an output directory
# to store the validation polygons
output_path = './Outputs/Validation_Images/validation_polygons'
if not os.path.exists(output_path):
    os.mkdir(output_path)
val_polygons_path = os.path.join(output_path,'polygons.shp')

# Set to True to overwrite the above file
replace_val_polygons = True


# Set to True to restore previous results.
load_pr_curve = False

# Else:
# Creating an output directory with the model ID
# to store the precision-recall curve
output_path = './Outputs/Validation_Images/precision_recall_curve'
if not os.path.exists(output_path):
    os.mkdir(output_path)
output_path = os.path.join(output_path, weights_id)
if not os.path.exists(output_path):
    os.mkdir(output_path)
precision_recall_curve_path = os.path.join(output_path,'precision_recall_curve.pkl')

# //-----------------------------------------------------------------------------------\\

# //-----------------------------------------------------------------------------------\\
# //------------------------------ Local Functions ------------------------------------\\

# //-----------------------------------------------------------------------------------\\

def get_reference_coordinate(im_path):
    '''Helper function to retrieve the image center from csvfile'''
    global validation_images_info    
    image_name = os.path.split(im_path)[1]
    if not re.findall(data_suffix, image_name):
        image_name = image_name.replace(labels_suffix, data_suffix)
        image_name = image_name.replace(prediction_suffix, data_suffix)

    mask = validation_images_info.filename.apply(lambda x: re.findall(image_name, x))
    mask = mask.apply(lambda x: len(x) > 0)
    center = validation_images_info[[center_long,center_lat]][mask].values[0]
    return center

# //-----------------------------------------------------------------------------------\\

def extract_polygons(idx, flag='validation', thr_decision=0.9):
    '''Pipeline for extract the polygons in one image'''
    global validation_images, prediction_images
    
    # Get the image
    if flag == 'validation':
        im = validation_images[idx][0]
        im_path = validation_images.batch_ims_paths[0]

    elif flag == 'prediction':
        im = prediction_images[idx][0]
        im_path = prediction_images.batch_ims_paths[0]

    # Get the reference center
    center = get_reference_coordinate(im_path)

    # Fit the polygon extractor
    input_parameters = {'im' : im,
                        'center' : center,
                        'crs_rep' : crs_rep,
                        'zoom' : zoom,
                        'thr_decision' : thr_decision}

    p = rp.PolygonsExtractor(**input_parameters)

    if idx == 0 and flag=='validation':
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

def metrics_by_thr(thr_decision=0.9):
    '''Computes metrics by threshold decision'''
    global val_polygons, prediction_images
    
    print(f'Current threshold decision: {thr_decision:.2f}')

    # Return the predicted polygons by given threshold
    pred_polygons = [extract_polygons(idx, 'prediction', thr_decision=thr_decision)
                        for idx in range(len(prediction_images))]
    pred_polygons = np.concatenate(pred_polygons)
    pred_polygons = rp.dissolve_polygons(pred_polygons)
    pred_polygons = pred_polygons.to_crs(crs_rep)


    # Get metrics
    _, _, tp, fp, fn, d, displacement = evf.prediction_metrics(val_polygons, 
                                                               pred_polygons)
    
    tp_areas = pred_polygons.iloc[tp].area.values
    fp_areas = pred_polygons.iloc[fp].area.values
    fn_areas = val_polygons.iloc[fn].area.values

    # Return metrics
    metrics = {'tp_areas': tp_areas,
               'fp_areas': fp_areas,
               'fn_areas': fn_areas,
               'dice_coeff': d, 
               'displacement': displacement}

    return metrics

# //-----------------------------------------------------------------------------------\\

def precision_recall_by_thr(metrics):
    '''Retrives precision and recall for all thresholds'''

    stats = [evf.get_stats(i) for i in metrics]
    
    tps = np.concatenate([i[0].Total.values for i in stats])
    fps = np.concatenate([i[1].Total.values for i in stats])
    fns = np.concatenate([i[2].Total.values for i in stats])
    
    recall = evf.recall(tps, fns)
    precision = evf.precision(tps, fps)
    
    return precision, recall

# //-----------------------------------------------------------------------------------\\

# //-----------------------------------------------------------------------------------\\
# //-----------------------------------  Run  -----------------------------------------\\

if __name__ == '__main__':

    init = time.time()
    
    # //-------------------------------------------------------------------------------\\

    # Loading

    # Validation labels
    validation_labels_paths = files_paths(validation_images_path, 
                                          nested_carpets=True, exts=labels_suffix)
    input_parameters = {'ims_paths' : validation_labels_paths,
                        'batch_size' : batch_size,
                        'im_prep_funcs' : val_prep_funcs,
                        'im_prep_params' : val_prep_params}
    validation_images = DataGenerator(**input_parameters)


    # Prediction probability masks
    predictions_paths = [i.replace(validation_images_path, 
                                   predictions_path) for i in validation_labels_paths]
    predictions_paths = [i.replace(labels_suffix, 
                                   prediction_suffix) for i in predictions_paths]

    input_parameters = {'ims_paths' : predictions_paths,
                        'batch_size' : batch_size,                        
                        'im_prep_funcs' : pred_prep_funcs,
                        'im_prep_params' : pred_prep_params}
    prediction_images = DataGenerator(**input_parameters)

    print('\nEvaluating on:')
    print(validation_images)


    # //-------------------------------------------------------------------------------\\
    
    # Loading the images center coordinates
    validation_images_info = pd.read_csv(validation_images_info)

    # //-------------------------------------------------------------------------------\\

    # Extracting polygons from validation set
    if not os.path.exists(val_polygons_path) or replace_val_polygons:

        print('\nExtracting validation polygons')
        # Parallel processing
        pool = mp.Pool(processes=n_cores)
        val_polygons = pool.map(extract_polygons, range(len(validation_images)))
        pool.close()
    
        # Dissolve
        val_polygons = np.concatenate(val_polygons)
        val_polygons = rp.dissolve_polygons(val_polygons)
        val_polygons = val_polygons.to_crs(crs_rep)
        val_polygons.to_file(driver='ESRI Shapefile', filename=val_polygons_path)
        print(f'\nSaving validation polygons at: {val_polygons_path}')
    else:
        print(f'Loading validation polygons from: {val_polygons_path}')
        val_polygons = gpd.read_file(val_polygons_path)
        val_polygons = val_polygons.to_crs(crs_rep)

    # //-------------------------------------------------------------------------------\\

    # Extracting polygons in predictions

    # Process in parallel for each threshold decision
    if not load_pr_curve:
        print('\nExtracting predicted polygons')
        pool = mp.Pool(processes=n_cores)
        metrics = pool.map(metrics_by_thr, thresholds)
        pool.close()
        save_pickle(metrics, precision_recall_curve_path)
        print(f'\nSaving metrics at {precision_recall_curve_path}')
    else:
        print(f'Loading metrics from: {precision_recall_curve_path}')
        metrics = load_pickle(precision_recall_curve_path)

    # //-------------------------------------------------------------------------------\\

    # Get best score
    precision, recall = precision_recall_by_thr(metrics)
    opt_thr = evf.find_best_thr_decision(recall, precision)
    m = metrics[opt_thr]
    tp_stats, fp_stats, fn_stats = evf.get_stats(m)
    f1_score = evf.f1_score(recall[opt_thr], precision[opt_thr])

    print('\n')
    print('//--------------------------------------------------------\\')
    print('\nBest metrics:\n')
    print(f'Threshold decision : {thresholds[opt_thr]:.2f}')
    print(f'Dice coefficient: {tp_stats["DC"].values[0]:.4f}')
    print(f'Recall: {recall[opt_thr]:.4f}')
    print(f'Precision: {precision[opt_thr]:.4f}')
    print(f'F1 score: {f1_score:.4f}')
    print('\n')
    print(tp_stats)
    print('\n')
    print(fp_stats)
    print('\n')
    print(fn_stats)
    print('\n')
    print('//--------------------------------------------------------\\')
    print('\n')
    
    evf.plot_precision_recall_curve(precision, recall, multiline=False, labels=None)
    evf.plot_dc_tp_fp_fn(m)

    # //-------------------------------------------------------------------------------\\

    # Thresholding by area
    precisions = []
    recalls = []
    labels = []

    for thr_area in thr_areas:
        # Filtering polygons by threshold area
        metrics_by_thr_area = [evf.thresholding_by_area(m, thr_area) for m in metrics]
        
        # Get best score
        precision, recall = precision_recall_by_thr(metrics_by_thr_area)
        opt_thr = evf.find_best_thr_decision(recall, precision)
        m = metrics_by_thr_area[opt_thr]
        tp_stats, fp_stats, fn_stats = evf.get_stats(m)
        f1_score = evf.f1_score(recall[opt_thr], precision[opt_thr])
        omitted_area = val_polygons[val_polygons.area > thr_area].area.sum() / val_polygons.area.sum()

        print('\n')
        print('//--------------------------------------------------------\\')
        print('\nBest metrics:\n')
        print(f'Threshold area: {thr_area} m²')
        print(f'Omitted area: {100-omitted_area*100:.0f} %')
        print(f'Threshold decision: {thresholds[opt_thr]:.2f}')
        print(f'Dice coefficient: {tp_stats["DC"].values[0]:.4f}')
        print(f'Recall: {recall[opt_thr]:.4f}')
        print(f'Precision: {precision[opt_thr]:.4f}')
        print(f'F1 score: {f1_score:.4f}')
        print('\n')
        print(tp_stats)
        print('\n')
        print(fp_stats)
        print('\n')
        print(fn_stats)
        print('\n')
        print('//--------------------------------------------------------\\')
        print('\n')

        evf.plot_precision_recall_curve(precision, recall, multiline=False, labels=None)
        evf.plot_dc_tp_fp_fn(m)

        precisions.append(precision)
        recalls.append(recall)
        labels.append(f'Thr area : {thr_area} m²')


    precisions = np.array(precisions)
    recalls = np.array(recalls)
    evf.plot_precision_recall_curve(precisions, recalls, multiline=True, labels=labels)

    # //-------------------------------------------------------------------------------\\

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