# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:44:00 2020

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


def generate_negatives():
    # filepath = r"raster/raster_roads.tif"
    filepath = r"road_buffer.tif"
    
    # Open the file:
    raster = gdal.Open(filepath)
    
    # # Projection
    # projection = raster.GetProjection()
    
    # # Dimensions
    # XSize = raster.RasterXSize
    # YSize = raster.RasterYSize
    
    # # Number of bands
    # number_bands = raster.RasterCount
    
    # # Metadata for the raster dataset
    # metadata = raster.GetMetadata()
    
    # # Read the raster band as separate variable
    # band = raster.GetRasterBand(1)
    # NoData_value = 0
    # band.SetNoDataValue(NoData_value)
    # band.FlushCache()
    
    # # Check type of the variable 'band'
    # type(band)
    
    # # Data type of the values
    # data_type = gdal.GetDataTypeName(band.DataType)
    
    # Read raster data as numeric array from GDAL Dataset
    rasterArray = raster.ReadAsArray()
    # print(np.sum(rasterArray))
    
    
    """ PARAMETERS START """  
    fraction_train = 0.8
    fraction_valid = 0.1
    fraction_test  = 0.1
    year = 1940
    randomize = True
    
    offset = 0
    
    scale = "25"
    feature_type = 'roads'
    """ PARAMETERS STOP """
    schema = {
        'geometry': "Point",
        'properties': {"scale": "str", "type": "str", "year": "str", "origin": "str"},
    }
    min_samples = 15000
    sampling_fraction = 10000
    num_nonroad_points = 0
    resolution = 1.25
    # x_min = 677490.628122
    # y_max = 254000.316701
    # the spatial difference between negative and positive points (left & top)
    x_min = 676561.699299 
    y_max = 254012.102560
    
    
    roads_dir = "training/roads"
    out_shp = roads_dir + "/negative_points.shp"
    
    with fiona.open(out_shp, 'w', 'ESRI Shapefile', schema) as c:
        rasterArray = np.where(rasterArray > 0.5, 0, 1)
        #acquire coordinates of white points(road sgment)
        feature_coordinates = np.argwhere(rasterArray == 0)
        #determine the number of points, depends on sampling fraction
        num_random_points = max(int(np.sum(rasterArray) / sampling_fraction), min_samples)
        #shuffle the indices
        indices = np.arange(len(feature_coordinates))
        np.random.shuffle(indices)
        #select the first num_random_points elements in the shuffled incices array
        indices_reduced = indices[:num_random_points]
        #print(len(indices), len(indices_reduced))
        
        feature_coordinates_reduced = feature_coordinates[indices_reduced]
        #print(feature_coordinates_reduced)
        #transform the coordinates form pixel to point
        feature_geocoordinates = feature_coordinates_reduced.copy()
        feature_geocoordinates[:,[0, 1]] = feature_geocoordinates[:,[1, 0]]
        feature_geocoordinates[:,0] = feature_geocoordinates[:,0] * resolution + x_min
        feature_geocoordinates[:,1] = feature_geocoordinates[:,1] * -resolution + y_max
        # feature_geocoordinates[:,0] = feature_geocoordinates[:,0] * resolution 
        # feature_geocoordinates[:,1] = feature_geocoordinates[:,1] * -resolution
        
        #print(feature_geocoordinates)
        
        # prepare for sampling  
        type_list = []
        for i in range(len(feature_geocoordinates)):
            val = random.random()
            Ttype = "training"
            if val > (fraction_train + fraction_valid):
                Ttype = "testing"
            elif val > fraction_train:
                Ttype = "validation"
            else:
                Ttype = "training"
            
            type_list.append(Ttype)
        
        
        # for p_coords in feature_geocoordinates:
        for i in range(len(feature_geocoordinates)):
            p_coords = feature_geocoordinates[i]
            Ttype = type_list[i]
            p = Point(p_coords[0], p_coords[1])
            feature = {"properties" : {"scale": scale, "year": year,"type":Ttype, "origin": feature_type}, "geometry": mapping(p)}
            c.write(feature)
            num_nonroad_points += 1
    print("Generated negative points successfully!")



def generate_positives():
    
    """ PARAMETERS START """
    multipliers = {"roads": 1}
    
    fraction_train = 0.8
    fraction_valid = 0.1
    fraction_test  = 0.1
    
    randomize = True
    
    offset = 12
    """ PARAMETERS STOP """
    
    base_dir = "training"
    out_location = "training/roads/positive_points.shp"
    
    feature_types = os.listdir(base_dir)
    
    scale = "25"
    
    # collect features
    features_dict = {}
    
    for feature_type in feature_types:
        feature_type_dir = base_dir + "/" + feature_type
        features_dict[feature_type] = []
        
        # years = os.listdir(feature_type_dir)
        # # years = os.walk(feature_type_dir)
        # for year in years:
        #     year_dir = feature_type_dir + "/" + year
        #     points_location = year_dir + "/points.shp"
    
        #     points = fiona.open(points_location)
        #     point_features = [{"type": point["type"], "id": point["id"], "properties":  {"scale": scale, "year": year, "origin": feature_type}, "geometry" : point["geometry"]} for point in points]
            
        #     features_dict[feature_type] += point_features
        
        year = "1940"
        year_dir = feature_type_dir + "/" + year
        points_location = year_dir + "/points.shp"

        points = fiona.open(points_location)
        point_features = [{"type": point["type"], "id": point["id"], "properties":  {"scale": scale, "year": year, "origin": feature_type}, "geometry" : point["geometry"]} for point in points]
        
        features_dict[feature_type] += point_features
    
            
    # shuffle by feature type
    for key, _ in features_dict.items():
        random.shuffle(features_dict[key])
        
     
    
    # multiply and aggregate
    features_aggregated = []
    for k, features in features_dict.items():
        m = multipliers[k]
        
        if m > 1:
            features_aggregated += features * int(m)
        elif 0 < m :
            features_aggregated += features[:int(len(features)*m)]
        else:
            print("WARNING: Multiplier invalid. Defaulting to 1.")
            features_aggregated += features
            
        
        
    # prepare for sampling
    random.shuffle(features_aggregated)
    
    for feature in features_aggregated:
        val = random.random()
       
        if val > (fraction_train + fraction_valid):
            type = "testing"
        elif val > fraction_train:
            type = "validation"
        else:
            type = "training"
        
        feature["properties"]["type"] = type
        
        if randomize:
            coords = shape(feature["geometry"]).coords
            x_offset = random.random() * offset - offset / 2
            y_offset = random.random() * offset - offset / 2
            x = coords[0][0] + x_offset
            y = coords[0][1] + y_offset
            
            feature["geometry"] = mapping(Point(x, y))
    
    
    schema = {
        'geometry': "Point",
        'properties': {"scale": "str", "type": "str", "year": "str", "origin": "str"},
    }
    
    
    with fiona.open(out_location, 'w', 'ESRI Shapefile', schema) as c:
                for feature in features_aggregated:
                    c.write(feature)
    
    print("Generated positive points successfully!")
    
def merge_positives_negatives(positive,negative):
    schema = {
        'geometry': "Point",
        'properties': {"scale": "str", "type": "str", "year": "str", "origin": "str"},
    }
    
    output_dir_roads = "training/roads"
    out_shp = output_dir_roads + "/points_aggregated.shp"
    files = [positive,negative]
    with fiona.open(out_shp, 'w', 'ESRI Shapefile', schema, crs=from_epsg(21781)) as c:
        for file in files:
            print(file)
            features = fiona.open(file)
            
            for feature in features:
                if feature["geometry"] is None:
                    print("WARNING: Feature is empty.")
                    continue
                               
                # feature["properties"].clear()
                # # feature["geometry"] = mapping(shape(feature["geometry"]).buffer(0))                    
                c.write(feature)
    print("Merged positive and negative points successfully!")

def extract_points_within_polygon(points_path, polygon_path,size):
    '''
    

    Parameters
    ----------
    points_path : TYPE
        DESCRIPTION.
    polygon_path : TYPE
        DESCRIPTION.
    size : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    polygon = fiona.open(polygon_path)
    point_features = fiona.open(points_path)  
    schema = {
        'geometry': "Point",
        'properties': {"scale": "str", "type": "str", "year": "str", "origin": "str"},
    }
    
    output_dir_roads = "training/roads"
    out_shp = output_dir_roads + "/points_aggregated_selected.shp"
    Num = 0
    # with fiona.open(out_shp, 'w', 'ESRI Shapefile', schema, crs=from_epsg(21781)) as c:
    #     for polyg_feature in polygon:
    #         eroded_polygon = polyg_feature['geometry']['coordinates'].buffer(size) 
    #         for feature in point_features:
    #             if eroded_polygon.contains(feature['geometry']['coordinates']):                                       
    #                 c.write(feature)
    #                 Num = Num + 1
    with fiona.open(out_shp, 'w', 'ESRI Shapefile', schema, crs=from_epsg(21781)) as c:
        for polyg_feature in polygon:
            cords_polygon = polyg_feature['geometry']['coordinates']
            eroded_polygon = Polygon(cords_polygon[0]).buffer(size) 
            for feature in point_features:
                cords_point = feature['geometry']['coordinates']
                point = Point(cords_point)
                if eroded_polygon.contains(point):                                       
                    c.write(feature)
                    Num = Num + 1
                # select 40000 points
                if Num == 40000:
                    break
    Num = str(Num)
    print("Selected "+Num+" aggregated points successfully!")
    
    


if __name__ == "__main__":
    ###Base parameters
    resolution = 1.25
    subimage_pixel = 128
    ###Step 1: generate positive and negative points
    generate_positives()
    generate_negatives()
    output_dir_roads = "training/roads"
    positive = output_dir_roads + "/positive_points.shp"
    negative = output_dir_roads + "/negative_points.shp"
    merge_positives_negatives(positive, negative)
    
    ###Step2: select aggregated points within polygon
    input_dir_roads = "training/roads"
    points_path = input_dir_roads + "/points_aggregated.shp"
    polygon_path = r"F:\corrected\25\1940\perimeter_Zurich.shp"
    buffer_size = math.sqrt(2)*resolution*subimage_pixel/2
    extract_points_within_polygon(points_path, polygon_path,-buffer_size)
    
    print('OK!')