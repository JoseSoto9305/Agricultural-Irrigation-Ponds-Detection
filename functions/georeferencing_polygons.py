#!/usr/bin/env python
# -*- coding: utf-8 -*-

__project__ = 'Agricultural Irrigation Ponds Detection'
__file__ = './functions/georeferencing_polygons.py'
__copyright__ = 'Copyright 2020, INIRENA-UMSNH'
__license__ = 'GPL'
__version__ = '1.0'
__date__ = 'March, 2020'
__maintainer__ = 'Jose Trinidad Soto Gonzalez'


'''

./functions/georeferencing_polygons.py


Functions to extract, and georeferencing polygons in an image


Defined classes:

    * ImageGeoreferencer
    * PolygonsExtractor

Defined functions:
    
    * dissolve_polygons

Required Third-party libraries:

    * geopandas
    * matplotlib
    * numpy
    * pandas
    * scipy
    * seaborn
    * shapely
    * skimage

Required custom modules:

    * ./functions/image

'''

# //-----------------------------------------------------------------------------------\\

# Import dependencies
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt

from scipy.sparse.csgraph import connected_components

from skimage import img_as_float
from skimage.morphology import label
from skimage.measure import find_contours

from shapely.geometry import Polygon
from shapely.geometry import Point

# Custom modules
sys.path.append('./functions')
from image import BaseImage

# //-----------------------------------------------------------------------------------\\

sns.set_style('darkgrid')

# //-----------------------------------------------------------------------------------\\

class ImageGeoreferencer(BaseImage):

    '''
    Parent class for georeferencing Google Map Satellite Imagery
    See ./georeferencing_polygons for a further implementation.
    '''    

    instance_id = 0

    def __init__(self, im,
                       center=(-101.00, 19.00),
                       crs_rep={'init':'epsg:32614'}, 
                       zoom=18):

        ImageGeoreferencer.instance_id += 1

        # Image dimentions
        self.h, self.w, _ = self._image_dim(im)
        
        # Image center coordinate in long, lat:
        self.center = gpd.GeoSeries([Point(center)], crs={'init':'epsg:4326'})

        # Image zoom
        self.zoom = zoom

        # Coordinate reference system to reproject the (long, lat) reference center
        self.crs = crs_rep

        # Image corners and pixel resolution
        self.corners = self.get_corners(close_polygon=True)
        self.resolution_x = (self.corners[1][0] - self.corners[0][0]) / self.w
        self.resolution_y = (self.corners[0][1] - self.corners[3][1]) / self.h


        # Information to display in the __repr__ and __str__ methods
        self.show_infos = ['Center', 'CRS', 'Pixel resolution X', 'Pixel resolution Y',
                            'Zoom image', 'ID instance', 'xmin',  'xmax', 'ymin', 'ymax']

        x,y = self.center.x[0], self.center.y[0]
        xmin, ymin = self.corners.min(axis=0)
        xmax, ymax = self.corners.max(axis=0)
        self.show_values = [(x,y), self.crs, self.resolution_x, self.resolution_y, 
                            self.zoom, self.instance_id, xmin, xmax, ymin, ymax]


    def get_corners(self, close_polygon=True):
        '''

        Retrieves image corners (x,y):

        [ 
          [upper left], 
          [upper right], 
          [bottom right], 
          [bottom left],
          [upper left]
        ] 

        Adapted from ishukshin's Github repository
        https://github.com/ishukshin/image-coordinates.py/blob/master/getcoords.py

        '''

        lng = self.center.x.values[0]
        lat = self.center.y.values[0]
        parallel_multiplier = np.cos(lat * np.pi / 180)
        degrees_per_pixel_x = 360 / np.power(2, self.zoom + 8)
        degrees_per_pixel_y = 360 / np.power(2, self.zoom + 8) * parallel_multiplier

        def get_point_lat_lng(x, y):
            point_long = lng + degrees_per_pixel_x * ( x  - self.w / 2)
            point_lat = lat - degrees_per_pixel_y * ( y - self.h / 2)            
            return point_long, point_lat

        if close_polygon:
            g = gpd.GeoSeries([Point(get_point_lat_lng(0, 0)),
                               Point(get_point_lat_lng(self.w, 0)),
                               Point(get_point_lat_lng(self.w, self.h)),
                               Point(get_point_lat_lng(0, self.h)),
                               Point(get_point_lat_lng(0, 0))], crs={'init':'epsg:4326'})
        else:
            g = gpd.GeoSeries([Point(get_point_lat_lng(0, 0)),
                               Point(get_point_lat_lng(self.w, 0)),
                               Point(get_point_lat_lng(self.w, self.h)),
                               Point(get_point_lat_lng(0, self.h))], crs={'init':'epsg:4326'})

        g = g.to_crs(self.crs)
        self.corners = np.column_stack((g.x, g.y))
        self.center = self.center.to_crs(self.crs)
        return self.corners

# //-----------------------------------------------------------------------------------\\    

class PolygonsExtractor(ImageGeoreferencer):
    
    '''
    Class for extract polygons in an image
    See ./polygons_extractor.py or ./evaluation.py for a further implementation.
    '''    

    # Empty dictionary to store the polygons
    polygons_dict = {}

    def __init__(self, 
                 im,
                 center=(-101.00, 19.00),
                 crs_rep={'init':'epsg:32614'}, 
                 zoom=18,
                 thr_decision=0.9,
                 reset_polygons_dict=True):

        ImageGeoreferencer.__init__(self, 
                                    im=im,
                                    center=center,
                                    crs_rep=crs_rep,
                                    zoom=zoom)
        
        # Processing the input image
        img = im.copy()
        img = img_as_float(img)
        self.thr_decision = thr_decision        
        img = self._thr_decision(img, thr_decision)
        self._labels(img)

        # Information to display in the __repr__ and __str__ methods
        self.show_infos.append('Threshold decision')
        self.show_values.append(self.thr_decision)
        
        if reset_polygons_dict:
            self.instance_id = 1
            self.polygons_dict = {}

    
    def _thr_decision(self, im, thr_decision):
        '''
        Threshold decision. 
        This is for a pixel binary classification.
        '''
        im[im > thr_decision] = 1
        im[im <= thr_decision] = 0
        return im

    
    def _labels(self, im):
        self.labels = label(im)
        # Omit background labels
        self.polygons_index = [i for i in np.unique(self.labels) if not i == 0]
        return self.labels, self.polygons_index

    
    def get_polygons(self):
        '''
        Get the polygons in the georeferenced image.
           ** Retrives polygons in epsg 4326 proyection
        '''

        # Coordinate reference system to reproject the polygons
        crs = {'init':'epsg:4326'}

        # Upper left image coordinate
        ul = self.corners[0]
        
        # Detect polygon vertices in the image
        polygons = []
        for i in self.polygons_index:
            mask = np.zeros((self.h, self.w), dtype='uint8')
            mask[self.labels == i] =  1
            polygon = find_contours(mask, 0.8)
            polygons.extend(polygon)

        # Flip (y,x) to (x,y)
        polygons = [np.fliplr(i) for i in polygons]

        # Referencing the vertices into the image
        polygons = [np.array( (ul[0] + polygons[i][:,0] * self.resolution_x, 
                               ul[1] - polygons[i][:,1] * self.resolution_y)).T
                    for i in range(len(polygons))]
        
        # Update the polygons dictionary
        for i in range(len(polygons)):
            id_name = f'im-{self.instance_id}--Polygon--00{i}'
            serie = gpd.GeoSeries([Polygon(polygons[i])])
            serie.crs = self.crs
            self.polygons_dict[id_name] = serie.to_crs(crs)[0]
        return self.polygons_dict


    def plot_polygons(self):
        '''        
        Shows the extracted polygons        
        '''

        # Polygons are saved in long-lat coordinate system, 
        # Convert image corners to long-lat coordinate system
        crs = {'init':'epsg:4326'}

        corners = gpd.GeoSeries([Polygon(self.corners)])
        corners.crs = self.crs
        corners = corners.to_crs(crs)

        polygons = self.polygons2geopandas()

        fig, ax = plt.subplots(1,1, figsize=(8,6))

        polygons.plot(color='red', ax=ax)
        xmin = min(corners.geometry.iloc[0].exterior.xy[0])
        xmax = max(corners.geometry.iloc[0].exterior.xy[0])
        ymin = min(corners.geometry.iloc[0].exterior.xy[1])
        ymax = max(corners.geometry.iloc[0].exterior.xy[1])

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.locator_params(axis='y', nbins=4)
        ax.locator_params(axis='x', nbins=4)

        plt.show()            


    def polygons2geopandas(self):
        '''
        Converts the polygons_dict to a geopandas dataframe
            ** Retrives the geopandas dataframe in epsg 4326 proyection
        '''        
        crs = {'init':'epsg:4326'}
        columns = ['ID','Polygons']
        df = pd.DataFrame.from_records(list(self.polygons_dict.items()), 
                                       columns=columns)
        polygons = gpd.GeoDataFrame(df.ID, geometry=df.Polygons, crs=crs)
            
        return polygons

# //-----------------------------------------------------------------------------------\\

def dissolve_polygons(polygons):
    '''    
    Dissolves overlapping polygons
        ** Retrieves the geopandas dataframe in epsg 4326 proyection
    '''    

    crs = {'init':'epsg:4326'}

    if isinstance(polygons, dict):
        s = gpd.GeoSeries( list(polygons.values()) )
    elif isinstance(polygons, list) or isinstance(polygons, np.ndarray):
        s = gpd.GeoSeries(polygons)

    overlap_matrix = s.apply(lambda x: s.intersects(x)).values.astype(int)
    _, ids = connected_components(overlap_matrix, directed=False)
    gdf = gpd.GeoDataFrame({'geometry': s, 'ID': ids}, crs=crs)
    dissolved = gdf.dissolve(by='ID')
        
    return dissolved

# //-------------------------------------------------------------\\

# //-----------------------------------------------------------------------------------\\
# //--------------------------------------------------------------------------\\
# //-----------------------------------------------------------------\\
# //-----------------------------------------------------\\
# //-----------------------------------------\\
# //-----------------------------\\
# //----------------\\
# //------\\
# END