# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 21:51:53 2021

@author: delladmin
"""

import os
import math
from osgeo import gdal
import numpy as np
import random
import fiona
from shapely.geometry import mapping, shape, LineString, Polygon, Point
from fiona.crs import from_epsg
import geopandas as gpd


def generate_data_augmentation_points(points_path, fraction):
    
    """ PARAMETERS START """
    
    randomize = True
    
    offset = 12
    """ PARAMETERS STOP """
    
    base_dir = "training"
    out_location = "training/roads/points_aggregated_selected_da.shp"
    points = gpd.read_file(points_path)
    training_points = points[points['type']=='training']
    validation_points = points[points['type']=='validation']
    num_train = int(len(training_points)*fraction)
    num_validation = int(len(validation_points)*fraction)
    dg_training = training_points.iloc[0:num_train]
    dg_validation = validation_points.iloc[0:num_validation]
    # training_points['DA'] = False
    # validation_points['DA'] = False
    points['DA'] = False
    dg_training['DA'] = True
    dg_validation['DA'] = True
    points = points.append(dg_training)
    points = points.append(dg_validation)

    points.to_file(out_location)
    
    print("Generated data augmentation points successfully!")





if __name__ == "__main__":
    ###Base parameters
    resolution = 1.25
    subimage_pixel = 64

    ###Step1: select aggregated points within polygon
    input_dir_roads = "training/roads"
    points_path = input_dir_roads + "/points_aggregated_selected.shp"
    generate_data_augmentation_points(points_path, 0.3)
    
    print('OK!')