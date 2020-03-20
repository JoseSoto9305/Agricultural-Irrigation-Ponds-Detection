#!/usr/bin/env python
# -*- coding: utf-8 -*-

__project__ = 'Agricultural Irrigation Ponds Detection'
__file__ = './functions/evaluate_functions.py'
__copyright__ = 'Copyright 2020, INIRENA-UMSNH'
__license__ = 'GPL'
__version__ = '1.0'
__date__ = 'March, 2020'
__maintainer__ = 'Jose Trinidad Soto Gonzalez'


'''

./functions/evaluate_functions.py


Functions to evaluate and visualize the performance of a Model


Defined functions:
    
    * dice_coeff
    * find_best_thr_decision
    * f1_score
    * get_stats
    * plot_dc_tp_fp_fn
    * plot_precision_recall_curve
    * prediction_metrics
    * precision
    * recall
    * thresholding_by_area

Required Third-party libraries:

    * geopandas
    * matplotlib
    * numpy
    * pandas
    * seaborn

'''

# //-----------------------------------------------------------------------------------\\

# Import dependencies
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt

from geopandas.tools.overlay import _overlay_intersection

# //-----------------------------------------------------------------------------------\\

sns.set_palette('tab10')
sns.set_style('darkgrid')

# //-----------------------------------------------------------------------------------\\

def dice_coeff(intersection, x, y):
    '''    
                    2|A n B|
    dice score =   __________
                    |A| + |B|

    Reference:
    https://stats.stackexchange.com/questions/195006/is-the-dice-coefficient-the-same-as-accuracy/253992
    '''
    return (2 * intersection) / (x + y)

# //-----------------------------------------------------------------------------------\\

def recall(n_tp, n_fn):
    '''
    recall = true_positives / (true_positives + false_negatives)
    '''
    return n_tp / (n_tp + n_fn)

# //-----------------------------------------------------------------------------------\\

def precision(n_tp, n_fp):
    '''
    precision = true_positives / (true_positives + false_positives)
    '''
    return n_tp / (n_tp + n_fp)

# //-----------------------------------------------------------------------------------\\

def f1_score(recall, precision):
    '''
    f1 = (2 * precision * recall) / (precision + recall)
    '''
    return (2 * precision * recall) / (precision + recall)

# //-----------------------------------------------------------------------------------\\

def find_best_thr_decision(recall, precision):
    '''
    Finds the best threshold decision for a precision-recall curve.
    '''
    opt = (1,1)
    dist = np.sqrt((recall - opt[0])**2 + (precision - opt[1])**2)
    return dist.argmin()

# //-----------------------------------------------------------------------------------\\

def prediction_metrics(ground_truth, prediction):
    '''    
    Returns:
    
        true positive indexes; 
        false positive indexes; 
        false negative indexes; 
        dice coefficient;
        displacement error between the centroid of ground truth 
            and the centroid of the predicted polygon.

    '''

    gt = ground_truth.copy()
    pred = prediction.copy()

    # Add column with an index
    gt['Roi_ID'] = range(ground_truth.shape[0])
    pred['Roi_ID'] = range(prediction.shape[0])

    # The gpd.overlay function returns an error if there is not any
    # match between the two goepandas dataframes. This lines are from
    # the gpd.overlay function, and then we used the _overlay_intersection function
    gt[gt._geometry_column_name] = gt.geometry.buffer(0)
    pred[pred._geometry_column_name] = pred.geometry.buffer(0)

    # Intersected area between polygons
    #res_inter = gpd.overlay(pred, gt, how='intersection')
    res_inter = _overlay_intersection(pred, gt)

    if res_inter.empty:
        print('No intersection beetwen polygons')
        ids_gt = None
        ids_pred = None
        d = None
        tp = None
        fp = pred.Roi_ID.values.tolist()
        fn = gt.Roi_ID.values.tolist()
        displacement = None
    else:
        # This line is also from the gpd.overlay function
        res_inter.drop(['__idx1', '__idx2'], axis=1, inplace=True)   

        # Overlapping indexes
        ids_pred = res_inter.Roi_ID_1.values
        ids_gt = res_inter.Roi_ID_2.values

        # Dice Coefficient
        intersection = res_inter.geometry.area.values
        x = pred.area.iloc[ids_pred].values
        y = gt.area.iloc[ids_gt].values
        d = dice_coeff(intersection, x, y)

        # Displacement between centroids
        gt_centroids = np.column_stack((gt.iloc[ids_gt].centroid.x.values,
                                        gt.iloc[ids_gt].centroid.y.values))

        pred_centroids = np.column_stack((pred.iloc[ids_pred].centroid.x.values,
                                          pred.iloc[ids_pred].centroid.y.values))

        displacement = np.sqrt((gt_centroids[:,0]-pred_centroids[:,0])**2 + 
                               (gt_centroids[:,1]-pred_centroids[:,1])**2)

        # Checking if there are true positive duplicates
        # if so, keep only the polygon with the
        # greater dice coefficient, and then, mark
        # the rest ones as false positives
        # *** If there is an easy way, let me know :D
        if res_inter.shape[0] > 1:
            v = np.column_stack((ids_gt, ids_pred, d, displacement))
            v = v[np.lexsort((v[:,2],v[:,0]))]
            mask = np.zeros(v.shape[0], dtype=bool)
            for i in range(v.shape[0]):
                if i == v.shape[0]-1:
                    if v[i,0] != v[i-1,0]:
                        mask[i] = True
                    elif v[i,0] == v[i-1,0]:
                        mask[i] = True                        
                else:
                    if v[i,0] != v[i+1,0]:
                        mask[i] = True
            
            ids_gt = v[mask, 0]
            ids_pred = v[mask,1]
            d = v[mask,2]
            displacement = v[mask, 3]
        
        # True Positive indexes
        tp = ids_pred.astype(int).tolist()

        # False Positive indexes
        fp = [i for i in pred.Roi_ID.values if i not in ids_pred]

        # False Negative indexes
        fn = [i for i in gt.Roi_ID.values if i not in ids_gt]

    return ids_gt, ids_pred, tp, fp, fn, d, displacement

# //-----------------------------------------------------------------------------------\\

def get_stats(metrics):    
    '''
    Retrives stadistics from true positives, 
    false positives and false negatives.
    '''

    tp_areas = metrics['tp_areas']
    fp_areas = metrics['fp_areas']
    fn_areas = metrics['fn_areas']
    dc = metrics['dice_coeff']
    displacement = metrics['displacement']
    
    if tp_areas is not None and fn_areas is not None:
        n_rois = tp_areas.shape[0] + fn_areas.shape[0]
        n_tp = tp_areas.shape[0]
    elif tp_areas is not None and fn_areas is None:
        n_rois = tp_areas.shape[0]
        n_tp = tp_areas.shape[0]
    elif tp_areas is None and fn_areas is not None:
        n_rois = fn_areas.shape[0]
    
    # True positives
    tp_stats = None
    if tp_areas is not None:
        rows = ['True Positives']
        columns = ['Total', 'Percentage', 
                   'DC', 'DC std', 
                   'displacement', 'displacement std',
                   'area', ' area_std']
        
        # Dice coefficient
        dc_mean, dc_std = None, None
        if dc is not None:
            dc_mean, dc_std = dc.mean(), dc.std()

        # Displacement beetwen centroids
        disp_mean, disp_std = None, None
        if displacement is not None:
            disp_mean, disp_std = displacement.mean(), displacement.std()

        data = np.array([[n_tp, n_tp/n_rois, 
                          dc_mean, dc_std,
                          disp_mean, disp_std,
                          tp_areas.mean(), tp_areas.std()]])    
        tp_stats = pd.DataFrame(data, columns=columns, index=rows)

    # False Positives
    fp_stats = None
    if fp_areas is not None:
        n_fp = fp_areas.shape[0]
        rows = ['False Positives']
        columns = ['Total', 'Percentage', 'area', ' area_std']
        data = np.array([[n_fp, n_fp/n_tp, 
                          fp_areas.mean(), fp_areas.std()]])
        fp_stats = pd.DataFrame(data, columns=columns, index=rows)

    # False Negatives
    fn_stats = None
    if fn_areas is not None:
        n_nd = fn_areas.shape[0]
        rows = ['False Negatives']
        columns = ['Total', 'Percentage', 'area', ' area_std']

        data = np.array([[n_nd, n_nd/n_rois,
                          fn_areas.mean(), fn_areas.std()]])
        fn_stats = pd.DataFrame(data, columns=columns, index=rows)
    return tp_stats, fp_stats, fn_stats

# //-----------------------------------------------------------------------------------\\

def thresholding_by_area(metrics, thr=30.):
    '''
    Thresholding ground truth and predicted polygons by a threshold
    area criteria. 
    '''
    tp_areas = None
    fp_areas = None
    fn_areas = None
    dc = None
    displacement = None
    if metrics['tp_areas'] is not None:
        tp_areas = metrics['tp_areas'][metrics['tp_areas'] > thr]
    if metrics['fp_areas'] is not None:
        fp_areas = metrics['fp_areas'][metrics['fp_areas'] > thr]
    if metrics['fn_areas'] is not None:
        fn_areas = metrics['fn_areas'][metrics['fn_areas'] > thr]
    if metrics['dice_coeff'] is not None:
        dc = metrics['dice_coeff'][metrics['tp_areas'] > thr]
    if metrics['displacement'] is not None:
        displacement = metrics['displacement'][metrics['tp_areas'] > thr]
    return {'tp_areas':tp_areas, 
            'fp_areas':fp_areas, 
            'fn_areas':fn_areas, 
            'dice_coeff':dc, 
            'displacement':displacement}

# //-----------------------------------------------------------------------------------\\

def plot_dc_tp_fp_fn(metrics):
    '''
    Plots Dice coefficient, true positive areas,
    false positive areas and false negative areas
    '''

    fig = plt.figure()
    fig.set_size_inches(12,8)
    st = dict(marker='^', markeredgecolor='black', markerfacecolor='black')

    ax0 = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
    ax1 = plt.subplot2grid((3, 2), (0, 1))
    ax2 = plt.subplot2grid((3, 2), (1, 1))
    ax3 = plt.subplot2grid((3, 2), (2, 1))
    
    h1,bins1 = np.histogram(metrics['tp_areas'], bins=8)
    h2,bins2 = np.histogram(metrics['fp_areas'], bins=8)
    h3,bins3 = np.histogram(metrics['fn_areas'], bins=8)

    sns.boxplot(data=metrics['dice_coeff'], width=.5, 
                showmeans=True, meanprops=st, ax=ax0, color='#3175F1')
    sns.distplot(metrics['tp_areas'], bins=bins1, kde=False, ax=ax1, color='#F13131')
    sns.distplot(metrics['fp_areas'], bins=bins2, kde=False, ax=ax2, color='#3BB824')
    sns.distplot(metrics['fn_areas'], bins=bins3, kde=False, ax=ax3, color='#EB7B16')
    min_max = (min(h1.min(),h2.min(), h3.min()),
               max(h1.max(),h2.max(), h3.max()))

    ax1.set_ylim(min_max[0],min_max[1]+50)
    ax2.set_ylim(min_max[0],min_max[1]+50)
    ax3.set_ylim(min_max[0],min_max[1]+50)

    ax0.set_xlabel('True Positives')
    ax0.set_ylabel('Dice Coefficient')
    ax0.set_title('Dice Coefficient')
    ax1.set_ylabel('Frecuency')
    ax1.set_xlabel('Area m²')
    ax1.set_title('True Positives')
    ax2.set_ylabel('Frecuency')
    ax2.set_xlabel('Area m²')
    ax2.set_title('False Positives')
    ax3.set_ylabel('Frecuency')
    ax3.set_xlabel('Area m²')
    ax3.set_title('False Negatives')
    plt.tight_layout()
    plt.show()

# //-----------------------------------------------------------------------------------\\

def plot_precision_recall_curve(precision, recall, multiline=False, labels=None):
    '''
    Precision-Recall curve
    '''

    fig = plt.figure()
    fig.set_size_inches(10,6)

    colors = np.array(['r', 'g', 'b', 'y','m','c']).reshape(1,-1)
    colors = np.repeat(colors, 100, axis=0).flatten()

    if not multiline:                
        best_thr = find_best_thr_decision(recall, precision)
        plt.plot(recall, precision, color=colors[0], label='PR-Curve')
        b = np.array([[recall[best_thr],0],
                        [recall[best_thr], precision[best_thr]]])
        plt.plot(b[:,0], b[:,1], f'--{colors[0]}', linewidth=0.5)
        b = np.array([[0, precision[best_thr]],
                       [recall[best_thr], precision[best_thr]]])
        plt.plot(b[:,0], b[:,1], f'--{colors[0]}', linewidth=0.5, label='Best Thr Decision')

    else:
        if labels is None:
            labels = [None]*len(precision)
        for i in range(len(precision)):                    
            plt.plot(recall[i], precision[i], colors[i], label=labels[i])
            best_thr = find_best_thr_decision(recall[i], precision[i])
            b = np.array([[recall[i][best_thr],0],
                            [recall[i][best_thr], precision[i][best_thr]]])
            plt.plot(b[:,0], b[:,1], f'--{colors[i]}', linewidth=0.5)
            b = np.array([[0, precision[i][best_thr]],
                           [recall[i][best_thr], precision[i][best_thr]]])
            plt.plot(b[:,0], b[:,1], f'--{colors[i]}', linewidth=0.5)
        plt.plot([],[], '--k', label='Best Thr Decision')
    
    plt.xlim([recall.min() - 0.05, recall.max() + 0.05])
    plt.ylim([precision.min() - 0.05, precision.max() + 0.05])

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

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
