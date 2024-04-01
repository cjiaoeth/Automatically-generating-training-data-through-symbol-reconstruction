from osgeo import ogr, gdal
import uuid
import numpy as np
from keras import backend as K
import time, datetime
import json, os
# import edge_detection as ed

#vector_uri = QgsDataSourceUri()
#vector_uri.setConnection(databaseConfig["host"], str(databaseConfig["port"]), databaseConfig["name"],
#                         databaseConfig["user"], databaseConfig["password"])


#Keras
# def DiceLoss(targets, inputs, smooth=1e-6):
    
#     #flatten label and prediction tensors
#     inputs = K.flatten(inputs)
#     targets = K.flatten(targets)
    
#     intersection = K.sum(K.dot(targets, inputs))
#     dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
#     return 1 - dice

############Sobel loss #########

#this contains both X and Y sobel filters in the format (3,3,1,2)
#size is 3 x 3, it considers 1 input channel and has two output channels: X and Y
sobelFilter = K.variable([[[[1.,  1.]], [[0.,  2.]],[[-1.,  1.]]],
                      [[[2.,  0.]], [[0.,  0.]],[[-2.,  0.]]],
                      [[[1., -1.]], [[0., -2.]],[[-1., -1.]]]])

def expandedSobel(inputTensor):

    #this considers data_format = 'channels_last'
    inputChannels = K.reshape(K.ones_like(inputTensor[0,0,0,:]),(1,1,-1,1))
    #if you're using 'channels_first', use inputTensor[0,:,0,0] above

    return sobelFilter * inputChannels

def sobelLoss(yTrue,yPred):

    #get the sobel filter repeated for each input channel
    filt = expandedSobel(yTrue)

    #calculate the sobel filters for yTrue and yPred
    #this generates twice the number of input channels 
    #a X and Y channel for each input channel
    sobelTrue = K.depthwise_conv2d(yTrue,filt,padding='same')
    sobelPred = K.depthwise_conv2d(yPred,filt,padding='same')

    #now you just apply the mse:
    return K.mean(K.square(sobelTrue - sobelPred))





##############################



# def sobel_coef(y_true, y_pred, smooth=1):
#     """
#     Dice = (2*|X & Y|)/ (|X|+ |Y|)
#          =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
#     ref: https://arxiv.org/pdf/1606.04797v1.pdf
#     """
#     y_trueS = ed.Sobel(y_true)
#     y_predS = ed.Sobel(y_pred)
#     intersection = K.sum(K.abs(y_trueS * y_predS), axis=-1)
#     return (2. * intersection + smooth) / (K.sum(K.square(y_trueS),-1) + K.sum(K.square(y_predS),-1) + smooth)

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def DiceLoss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def CustomDiceLoss(y_true, y_pred):
    dice = 1-dice_coef(y_true, y_pred)
    # sobel_dice = 1-sobel_coef(y_true, y_pred)
    sobel_dice = sobelLoss(y_true,y_pred)
    return (dice+sobel_dice)/2

def CustomSobelMseLoss(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    sobel_mse = sobelLoss(y_true,y_pred)
    return (mse+sobel_mse)/2

def raise_error(message):
    print("ERROR: " + message)


def get_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')



def ndshp2dict(shp_vector_layer, filter_func, format="Esri Shapefile"):
    nd_definition = shp_vector_layer.definition
    location = shp_vector_layer.get_location()

    training_driver = ogr.GetDriverByName(format)
    training_data_source = training_driver.Open(str(location))
    training_layer = training_data_source.GetLayer()

    point_layer = training_data_source.GetLayer()
    feature_count = training_layer.GetFeatureCount()

    points = []

    for i in range(feature_count):
        point = point_layer.GetNextFeature()
        if not filter_func(point):
            continue

        key_components = []
        for dim_field in nd_definition["dimension_fields"]:
            key_components.append(point.GetField(dim_field))

        key = "_".join(key_components)

        type = point.GetField('type')

        geometry = point.GetGeometryRef()
        x = geometry.GetX()
        y = geometry.GetY()

        points.append([x, y, type, key])


    sorted_points = sort_by_key(points, 3, True)

    return sorted_points




def ndshp2dict_DA(shp_vector_layer, filter_func, format="Esri Shapefile"):
    nd_definition = shp_vector_layer.definition
    location = shp_vector_layer.get_location()

    training_driver = ogr.GetDriverByName(format)
    training_data_source = training_driver.Open(str(location))
    training_layer = training_data_source.GetLayer()

    point_layer = training_data_source.GetLayer()
    feature_count = training_layer.GetFeatureCount()

    points = []

    for i in range(feature_count):
        point = point_layer.GetNextFeature()
        if not filter_func(point):
            continue

        key_components = []
        for dim_field in nd_definition["dimension_fields"]:
            key_components.append(point.GetField(dim_field))

        key = "_".join(key_components)

        type = point.GetField('type')
        DA = point.GetField('DA')

        geometry = point.GetGeometryRef()
        x = geometry.GetX()
        y = geometry.GetY()

        points.append([x, y, type, key, DA])


    sorted_points = sort_by_key(points, 3, True)

    return sorted_points

def trim_GeoJSON(data_location):

    with open(data_location) as json_file:
        points = json.load(json_file)["features"]

    points_trimmed = [[p["geometry"]["coordinates"][0],
                       p["geometry"]["coordinates"][1],
                       p["properties"]["type"],
                       p["properties"]["key"],
                       p["properties"]["sheet"],
                       p["properties"]["annot"]] for p in points]


    return points_trimmed

def trim_GeoJSON_DA(data_location):

    with open(data_location) as json_file:
        points = json.load(json_file)["features"]

    points_trimmed = [[p["geometry"]["coordinates"][0],
                       p["geometry"]["coordinates"][1],
                       p["properties"]["type"],
                       p["properties"]["DA"],
                       p["properties"]["key"],
                       p["properties"]["sheet"],
                       p["properties"]["annot"]] for p in points]


    return points_trimmed



def sort_by_key(points, idx, remove_key):
    points_sorted = {}

    for point in points:
        key = point[idx]

        if not key in points_sorted:
            points_sorted[key] = []

        if remove_key:
            del point[idx]

        points_sorted[key].append(point)


    return points_sorted


class NDFileManager:
    @staticmethod
    def file(path):
        with open(path) as nd_definition_raw:
            nd_definition = json.load(nd_definition_raw)
            nd_layer_type = nd_definition["type"]

            if nd_layer_type == "shp_vector_distributed":
                return NDDistributedSHPVectorLayer(path, nd_definition)

            elif nd_layer_type == "distributed_raster":
                return NDDistributedRasterLayer(path, nd_definition)

            elif nd_layer_type == "shp_vector":
                return NDSHPVectorLayer(path, nd_definition)



class NDDistributedSHPVectorLayer:
    def __init__(self, path, definition):
        self.definition = definition
        self.path = path
        self.dims = dict(zip(self.definition["dimensions"], self.definition["default_values"]))
        self.datasource_store = {}
        self.driver = ogr.GetDriverByName('ESRI Shapefile')

        self.location = self.definition["location"]
        if self.definition["relative"] == "True":
            self.location = os.path.dirname(self.path) + "/" + self.location

    def get_layer(self, key):
        self.dims = dict(zip(self.definition["dimensions"], key.split("_")))

        if not key in self.datasource_store:
            self.datasource_store[key] = self.driver.Open(self.replace_strings(self.location), 0)

        return self.datasource_store[key].GetLayer()

    def replace_strings(self, string):
        for dim_name, dim_value in self.dims.items():
            string = string.replace("$" + dim_name + "$", dim_value)

        return string


class NDDistributedRasterLayer:
    def __init__(self, path, definition):
        self.definition = definition
        self.path = path
        self.dims = dict(zip(self.definition["dimensions"], self.definition["default_values"]))

        self.location = self.definition["location"]
        if self.definition["relative"] == "True":
            self.location = os.path.dirname(self.path) + "/" + self.location

    def get_location(self, key):
        self.dims = dict(zip(self.definition["dimensions"], key.split("_")))
        return self.replace_strings(self.location)

    def replace_strings(self, string):
        for dim_name, dim_value in self.dims.items():
            string = string.replace("$" + dim_name + "$", dim_value)

        return string


class NDSHPVectorLayer:
    def __init__(self, path, definition):
        self.definition = definition
        self.path = path

        self.location = self.definition["location"]
        if self.definition["relative"] == "True":
            self.location = os.path.dirname(self.path) + "/" + self.location


    def get_location(self):
        return self.location
