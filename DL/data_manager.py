from osgeo import ogr, gdal
import os, uuid
import numpy as np
from timeit import default_timer as timer
import random
from sklearn.cluster import DBSCAN
import json
from utils import get_timestamp
import matplotlib.pyplot as plt
import cv2
import data_augmentation as DataAugmentation


        
                

class DataManager:
    """
    def __init__(self):
        self.databaseConfig = {
            "name": "siegfried",
            "host": "ikgstor.ethz.ch",
            "port": 5432,
            "database": "siegfried",
            "user": "postgres",
            "password": "storgres.732"
        }

        self.sheet_location = "Y:/Siegfried_Mosaics"

        conn_string = "PG: host=%s dbname=%s user=%s password=%s" % (
        self.databaseConfig["host"], self.databaseConfig["database"], self.databaseConfig["user"], self.databaseConfig["password"])

        self.connection = ogr.Open(conn_string)
        self.layer_store = {}

        print("Connection established.")

    def __del__(self):
        print("Connection destroyed.")
        self.connection.Destroy()


    def get_raster_location(self, key):
        scale, year = key.split("_")
        return self.sheet_location + "/" + str(scale) + "/" + str(scale) + "_" + str(year) + ".vrt"


    def get_vector(self, key, layer):
        scale, year = key.split("_")
        layer_key = layer + "." + scale + "_" + year
        if not layer_key in self.layer_store:
            self.layer_store[layer_key] = self.connection.GetLayerByName(layer + "." + scale + "_" + year)

        return self.layer_store[layer_key]

    """
    
    def __init__(self, raster_source, vector_source):
        self.rs = raster_source
        self.vs = vector_source
        
    
    @staticmethod
    def get_resolution(key):
        keys = key.split("_")
        scale = keys[0]
        if scale == "25":
            return 1.25
        else:
            return 2.5


    @staticmethod
    def crop_by_point(point, folder, in_size, out_size, upscale=False):
        source_patch = DataManager.crop_sheet_by_point(point, folder, in_size, upscale=upscale)
        target_patch = DataManager.crop_annotation_by_point(point, folder, out_size, upscale=upscale)

        return source_patch, target_patch

    def crop_by_point_new(point, folder, in_size, out_size, upscale=False):
        # source_patch = DataManager.crop_sheet_by_point(point, folder, in_size, upscale=upscale)
        # target_patch = DataManager.crop_annotation_by_point(point, folder, out_size, upscale=upscale)
        input_img, input_patch, input_left, input_top = DataManager.crop_sheet_by_point_new(point, folder, in_size, upscale=upscale)
        output_img, output_patch, input_leftT, input_topT = DataManager.crop_annotation_by_point_new(point, folder, out_size, upscale=upscale)
        tuple_source = (input_img, input_patch, input_left, input_top)
        touple_target = (output_img, output_patch, input_leftT, input_topT)
        return tuple_source, touple_target

    @staticmethod
    def crop_sheet_by_point(point, folder, input_size, normalize = True, upscale=False):
        x = point[0]
        y = point[1]
        type = point[2]
        key = point[3]
        sheet_location = str(folder + "/" + key + "/" + point[4])
        annotation_location = str(folder + "/" + key + "/" + point[5])

        resolution = DataManager.get_resolution(key)


        u = str(uuid.uuid4())
        cropped_sheet_location = '/vsimem/'+u+'_source.tif'

        if upscale:
            input_left = x - resolution * (input_size / 4)
            input_right = x + resolution * (input_size / 4)
            input_bottom = y - resolution * (input_size / 4)
            input_top = y + resolution * (input_size / 4)

            gdal.Translate(cropped_sheet_location, sheet_location, xRes=resolution/2, yRes=resolution/2, projWin=[input_left, input_top, input_right, input_bottom])

        else:
            input_left = x - resolution * (input_size / 2)
            input_right = x + resolution * (input_size / 2)
            input_bottom = y - resolution * (input_size / 2)
            input_top = y + resolution * (input_size / 2)

            gdal.Translate(cropped_sheet_location, sheet_location, projWin=[input_left, input_top, input_right, input_bottom])

        cropped_sheet_raster = gdal.Open(cropped_sheet_location)
        input_patch = cropped_sheet_raster.ReadAsArray()

        if len(input_patch.shape) == 2:
            input_patch = np.expand_dims(input_patch, -1)
        else:
            input_patch = np.moveaxis(input_patch, 0, -1)

        cropped_sheet_raster = None
        gdal.Unlink(cropped_sheet_location)

        if normalize:
            input_patch = input_patch / 255
            
        # plt.imshow(input_patch)
        # plt.show()
        return input_patch

    def crop_sheet_by_point_new(point, folder, input_size, normalize = True, upscale=False):
        x = point[0]
        y = point[1]
        type = point[2]
        key = point[4]
        sheet_location = str(folder + "/" + key + "/" + point[5])
        annotation_location = str(folder + "/" + key + "/" + point[6])

        resolution = DataManager.get_resolution(key)


        u = str(uuid.uuid4())
        cropped_sheet_location = '/vsimem/'+u+'_source.tif'

        if upscale:
            input_left = x - resolution * (input_size / 4)
            input_right = x + resolution * (input_size / 4)
            input_bottom = y - resolution * (input_size / 4)
            input_top = y + resolution * (input_size / 4)

            gdal.Translate(cropped_sheet_location, sheet_location, xRes=resolution/2, yRes=resolution/2, projWin=[input_left, input_top, input_right, input_bottom])

        else:
            input_left = x - resolution * (input_size / 2)
            input_right = x + resolution * (input_size / 2)
            input_bottom = y - resolution * (input_size / 2)
            input_top = y + resolution * (input_size / 2)

            gdal.Translate(cropped_sheet_location, sheet_location, projWin=[input_left, input_top, input_right, input_bottom])

        cropped_sheet_raster = gdal.Open(cropped_sheet_location)
        input_img = cropped_sheet_raster.ReadAsArray()
        input_patch = input_img.copy()
        if len(input_patch.shape) == 2:
            input_patch = np.expand_dims(input_patch, -1)
        else:
            input_patch = np.moveaxis(input_patch, 0, -1)

        cropped_sheet_raster = None
        gdal.Unlink(cropped_sheet_location)

        if normalize:
            input_patch = input_patch / 255
            
        # plt.imshow(input_patch)
        # plt.show()
        return input_img, input_patch, input_left, input_top

    @staticmethod
    def crop_annotation_by_point_new(point, folder, output_size, normalize = True, upscale=False):
        x = point[0]
        y = point[1]
        type = point[2]
        key = point[4]
        sheet_location = str(folder + "/" + key + "/" + point[5])
        annotation_location = str(folder + "/" + key + "/" + point[6])
        # ####set nodata as 0
        # target_ds = gdal.Open(annotation_location)
        # band = target_ds.GetRasterBand(1)
        # NoData_value = -999999
        # band.SetNoDataValue(NoData_value)
        # band.FlushCache()

        resolution = DataManager.get_resolution(key)

        u = str(uuid.uuid4())
        cropped_annotation_location = '/vsimem/' + u + '_target.tif'


        if upscale:
            input_leftT = x - resolution * (output_size / 4)
            input_rightT = x + resolution * (output_size / 4)
            input_bottomT = y - resolution * (output_size / 4)
            input_topT = y + resolution * (output_size / 4)

            gdal.Translate(cropped_annotation_location, annotation_location, xRes=resolution/2, yRes=resolution/2, projWin=[input_leftT, input_topT, input_rightT, input_bottomT])

        else:
            input_leftT = x - resolution * (output_size / 2)
            input_rightT = x + resolution * (output_size / 2)
            input_bottomT = y - resolution * (output_size / 2)
            input_topT = y + resolution * (output_size / 2)

            gdal.Translate(cropped_annotation_location, annotation_location, projWin=[input_leftT, input_topT, input_rightT, input_bottomT])



        cropped_annotation_raster = gdal.Open(cropped_annotation_location)


        output_image = cropped_annotation_raster.ReadAsArray()
        output_patch = output_image.copy()
        # plt.imshow(output_patch)
        # plt.show()
        if len(output_patch.shape) == 2:
            output_patch = np.expand_dims(output_patch, -1)
        else:
            output_patch = np.moveaxis(output_patch, 0, -1)

        cropped_annotation_raster = None
        gdal.Unlink(cropped_annotation_location)

        if normalize:
            output_patch = output_patch / 255


        return output_image, output_patch, input_leftT, input_topT

    def crop_annotation_by_point(point, folder, output_size, normalize = True, upscale=False):
        x = point[0]
        y = point[1]
        type = point[2]
        key = point[3]
        sheet_location = str(folder + "/" + key + "/" + point[4])
        annotation_location = str(folder + "/" + key + "/" + point[5])

        resolution = DataManager.get_resolution(key)

        u = str(uuid.uuid4())
        cropped_annotation_location = '/vsimem/' + u + '_target.tif'

        if upscale:
            input_left = x - resolution * (output_size / 4)
            input_right = x + resolution * (output_size / 4)
            input_bottom = y - resolution * (output_size / 4)
            input_top = y + resolution * (output_size / 4)

            gdal.Translate(cropped_annotation_location, annotation_location, xRes=resolution/2, yRes=resolution/2, projWin=[input_left, input_top, input_right, input_bottom])

        else:
            input_left = x - resolution * (output_size / 2)
            input_right = x + resolution * (output_size / 2)
            input_bottom = y - resolution * (output_size / 2)
            input_top = y + resolution * (output_size / 2)

            gdal.Translate(cropped_annotation_location, annotation_location, projWin=[input_left, input_top, input_right, input_bottom])



        cropped_annotation_raster = gdal.Open(cropped_annotation_location)


        output_patch = cropped_annotation_raster.ReadAsArray()
        # plt.imshow(output_patch)
        # plt.show()
        if len(output_patch.shape) == 2:
            output_patch = np.expand_dims(output_patch, -1)
        else:
            output_patch = np.moveaxis(output_patch, 0, -1)

        cropped_annotation_raster = None
        gdal.Unlink(cropped_annotation_location)

        if normalize:
            output_patch = output_patch / 255


        return output_patch


    @staticmethod
    def cluster_points(points, epsilon):
        db = DBSCAN(eps=epsilon, min_samples=1).fit(points)
        labels = db.labels_

        print(np.amin(labels), np.amax(labels))

        clusters = []

        max_idx = np.amax(labels)
        for a in range(max_idx + 1):
            cluster_indices = np.nonzero(labels == a)
            clusters.append(cluster_indices)

        return clusters



    @staticmethod
    def compute_bounding_box(cluster, padding):
        x_min = np.amin(cluster[:, 0])
        x_max = np.amax(cluster[:, 0])
        y_min = np.amin(cluster[:, 1])
        y_max = np.amax(cluster[:, 1])

        return([x_min - padding, y_max + padding, x_max + padding, y_min - padding])



    def extract_region(self, points, target_path, layers, padding, extract_labels=True, use_vrt=False):

        num_bands = len(layers)
        train_points = []
        for key, pnts in points.items():
            key_path = target_path + "/" + key
            os.makedirs(key_path)

            pnts_reduced = [pnt[0:2] for pnt in pnts]

            np_pnts_reduced = np.asarray(pnts_reduced)
            clusters = DataManager.cluster_points(np_pnts_reduced, 1000)

            resolution = DataManager.get_resolution(key)
            padding_scaled = resolution * padding

            cluster_counter = 0
            for cluster in clusters:
                #print(cluster)
                bbox = DataManager.compute_bounding_box(np_pnts_reduced[cluster[0]], padding_scaled)

                sheet_location_relative = str(cluster_counter).zfill(2) + "_sheet_crop.tif"
                sheet_location = str(key_path) + "/" + sheet_location_relative

                raster_location = self.rs.get_location(key)
                gdal.Translate(sheet_location, raster_location , projWin=bbox, format="GTiff")

                raster = gdal.Open(sheet_location)
                band = raster.GetRasterBand(1)
                width_px = band.XSize
                height_px = band.YSize
                geo_transform = raster.GetGeoTransform()

                suffix = ".tif"
                format = "GTiff"
                if use_vrt:
                    # note: a small test shows that using a VRT is approximately twice as slow as storing the data locally
                    # however, it is only a few KB in size in comparison to a few (up to several hundreds) of MB
                    suffix = ".vrt"
                    format= "VRT"

                if extract_labels:
                    rasterization_location_relative = str(cluster_counter).zfill(2) + "_annotations_crop" + suffix
                    rasterization_location = str(key_path) + "/" + rasterization_location_relative

                    target_ds = gdal.GetDriverByName(format).Create(rasterization_location, width_px, height_px, num_bands,
                                                                     gdal.GDT_Byte,
                                                                     options=["COMPRESS=PACKBITS"])

                    target_ds.SetGeoTransform((geo_transform[0], resolution, 0, geo_transform[3], 0, -resolution))

                    for o in range(1, num_bands+1):
                        layer = layers[o-1]

                        mb_l = self.vs.get_layer(key + "_" + layer)
                        band = target_ds.GetRasterBand(o)
                        NoData_value = -999999
                        band.SetNoDataValue(NoData_value)
                        band.FlushCache()

                        gdal.RasterizeLayer(target_ds, [o], mb_l)
                else:
                    rasterization_location_relative = ""


                for idx in cluster[0]:
                    pnt = pnts[idx]
                    train_points.append([pnt[0], pnt[1], pnt[2], key, sheet_location_relative, rasterization_location_relative])

                cluster_counter += 1

        """
        point_location_relative = str(target_path) + "/points.json"
        with open(point_location_relative, "w") as f:
            json.dump(train_points, f, indent=4)
        """


        point_location_relative_geojson = str(target_path) + "/points.geojson"
        driver = ogr.GetDriverByName("GeoJSON")
        datasource = driver.CreateDataSource(point_location_relative_geojson)
        layer = datasource.CreateLayer("", None, ogr.wkbPoint)

        layer.CreateField(ogr.FieldDefn("key", ogr.OFTString))
        layer.CreateField(ogr.FieldDefn("sheet", ogr.OFTString))
        layer.CreateField(ogr.FieldDefn("annot", ogr.OFTString))
        layer.CreateField(ogr.FieldDefn("type", ogr.OFTString))

        for train_point in train_points:
            definition = layer.GetLayerDefn()
            feature = ogr.Feature(definition)
            feature.SetField("type", train_point[2])
            feature.SetField("key", train_point[3])
            feature.SetField("sheet", train_point[4])
            feature.SetField("annot", train_point[5])

            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(train_point[0], train_point[1])

            feature.SetGeometry(point)
            layer.CreateFeature(feature)
            feat = geom = None

        datasource = layer = None


        """
        point_location_relative_shp = u + "_points.json"
        point_location_shp = str(location / point_location_relative_shp)

        driver =  ogr.GetDriverByName("Esri Shapefile")
        datasource = driver.CreateDataSource(point_location_shp)
        layer = datasource.CreateLayer("", None, ogr.wkbPoint)

        layer.CreateField(ogr.FieldDefn("key", ogr.OFTString))
        layer.CreateField(ogr.FieldDefn("sheet", ogr.OFTString))
        layer.CreateField(ogr.FieldDefn("annot", ogr.OFTString))
        layer.CreateField(ogr.FieldDefn("type", ogr.OFTString))

        for train_point in train_points:
            definition = layer.GetFieldDefn()
            feature = ogr.Feature(definition)
            feature.SetField("key", train_point[2])
            feature.SetField("sheet", train_point[3])
            feature.SetField("annot", train_point[4])

            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(train_point[0], train_point[1])

            feature.SetGeometry(point)
            layer.CreateFeature(feature)
            feat = geom = None

        datasource = layer = None
        """




    def extract_region_DA(self, points, target_path, layers, padding, extract_labels=True, use_vrt=False):

        num_bands = len(layers)
        train_points = []
        for key, pnts in points.items():
            key_path = target_path + "\\" + key
            os.makedirs(key_path)

            pnts_reduced = [pnt[0:2] for pnt in pnts]

            np_pnts_reduced = np.asarray(pnts_reduced)
            clusters = DataManager.cluster_points(np_pnts_reduced, 1000)

            resolution = DataManager.get_resolution(key)
            padding_scaled = resolution * padding

            cluster_counter = 0
            for cluster in clusters:
                #print(cluster)
                bbox = DataManager.compute_bounding_box(np_pnts_reduced[cluster[0]], padding_scaled)

                sheet_location_relative = str(cluster_counter).zfill(2) + "_sheet_crop.tif"
                sheet_location = str(key_path) + "/" + sheet_location_relative

                raster_location = self.rs.get_location(key)
                gdal.Translate(sheet_location, raster_location , projWin=bbox, format="GTiff")

                raster = gdal.Open(sheet_location)
                band = raster.GetRasterBand(1)
                width_px = band.XSize
                height_px = band.YSize
                geo_transform = raster.GetGeoTransform()

                suffix = ".tif"
                format = "GTiff"
                if use_vrt:
                    # note: a small test shows that using a VRT is approximately twice as slow as storing the data locally
                    # however, it is only a few KB in size in comparison to a few (up to several hundreds) of MB
                    suffix = ".vrt"
                    format= "VRT"

                if extract_labels:
                    rasterization_location_relative = str(cluster_counter).zfill(2) + "_annotations_crop" + suffix
                    rasterization_location = str(key_path) + "\\" + rasterization_location_relative

                    target_ds = gdal.GetDriverByName(format).Create(rasterization_location, width_px, height_px, num_bands,
                                                                     gdal.GDT_Byte,
                                                                     options=["COMPRESS=PACKBITS"])

                    target_ds.SetGeoTransform((geo_transform[0], resolution, 0, geo_transform[3], 0, -resolution))

                    for o in range(1, num_bands+1):
                        layer = layers[o-1]

                        mb_l = self.vs.get_layer(key + "_" + layer)
                        band = target_ds.GetRasterBand(o)
                        NoData_value = -999999
                        band.SetNoDataValue(NoData_value)
                        band.FlushCache()

                        gdal.RasterizeLayer(target_ds, [o], mb_l)
                else:
                    rasterization_location_relative = ""


                for idx in cluster[0]:
                    pnt = pnts[idx]
                    train_points.append([pnt[0], pnt[1], pnt[2], pnt[3], key, sheet_location_relative, rasterization_location_relative])

                cluster_counter += 1

        """
        point_location_relative = str(target_path) + "/points.json"
        with open(point_location_relative, "w") as f:
            json.dump(train_points, f, indent=4)
        """


        point_location_relative_geojson = str(target_path) + "/points.geojson"
        driver = ogr.GetDriverByName("GeoJSON")
        datasource = driver.CreateDataSource(point_location_relative_geojson)
        layer = datasource.CreateLayer("", None, ogr.wkbPoint)

        layer.CreateField(ogr.FieldDefn("key", ogr.OFTString))
        layer.CreateField(ogr.FieldDefn("sheet", ogr.OFTString))
        layer.CreateField(ogr.FieldDefn("annot", ogr.OFTString))
        layer.CreateField(ogr.FieldDefn("type", ogr.OFTString))
        layer.CreateField(ogr.FieldDefn("DA", ogr.OFTString))

        for train_point in train_points:
            definition = layer.GetLayerDefn()
            feature = ogr.Feature(definition)
            feature.SetField("type", train_point[2])
            feature.SetField("DA", train_point[3])
            feature.SetField("key", train_point[4])
            feature.SetField("sheet", train_point[5])
            feature.SetField("annot", train_point[6])


            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(train_point[0], train_point[1])

            feature.SetGeometry(point)
            layer.CreateFeature(feature)
            feat = geom = None

        datasource = layer = None


        """
        point_location_relative_shp = u + "_points.json"
        point_location_shp = str(location / point_location_relative_shp)

        driver =  ogr.GetDriverByName("Esri Shapefile")
        datasource = driver.CreateDataSource(point_location_shp)
        layer = datasource.CreateLayer("", None, ogr.wkbPoint)

        layer.CreateField(ogr.FieldDefn("key", ogr.OFTString))
        layer.CreateField(ogr.FieldDefn("sheet", ogr.OFTString))
        layer.CreateField(ogr.FieldDefn("annot", ogr.OFTString))
        layer.CreateField(ogr.FieldDefn("type", ogr.OFTString))

        for train_point in train_points:
            definition = layer.GetFieldDefn()
            feature = ogr.Feature(definition)
            feature.SetField("key", train_point[2])
            feature.SetField("sheet", train_point[3])
            feature.SetField("annot", train_point[4])

            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(train_point[0], train_point[1])

            feature.SetGeometry(point)
            layer.CreateFeature(feature)
            feat = geom = None

        datasource = layer = None
        """


    @staticmethod
    def assign_point_types(points):


        pass

    @staticmethod
    def create_point_train_generator(points, folder, batch_size, input_size, output_size, upscale=False):
        feature_count = len(points)

        feature_counter = 0
        while True:
            source_images = []
            target_images = []

            while len(source_images) < batch_size:
                if feature_counter == (feature_count-1):
                    feature_counter = 0

                point = points[feature_counter]
                feature_counter += 1

                source_patch, target_patch = DataManager.crop_by_point(point, folder, input_size, output_size, upscale=upscale)

                # plt.imshow(source_patch)
                # plt.show()
                # target_patch_copy = target_patch[:,:,0]
                # plt.imshow(target_patch_copy)
                # plt.show()

                #print(np.amin(source_patch), np.amax(source_patch), np.amin(target_patch), np.amax(target_patch))

                """
                cv2.imshow("source", source_patch)
                cv2.imshow("target", target_patch)
                cv2.waitKey()
                """

                """
                print(np.amin(source_patch), np.amax(source_patch))
                print(np.amin(target_patch), np.amax(target_patch))
                """

                """
                u = str(uuid.uuid4())
                source_patch_shifted = source_patch[:, :, [2,1,0]]
                cv2.imwrite("Z:/_Sandbox/train_dump/" + u + "source.png", source_patch_shifted * 255)
                cv2.imwrite("Z:/_Sandbox/train_dump/" + u + "target.png", target_patch * 255)
                """


                source_images.append(source_patch)
                target_images.append(target_patch)

            sources_array = np.asarray(source_images)
            targets_array = np.asarray(target_images)

            yield(sources_array, targets_array)
            # return(sources_array, targets_array)

    @staticmethod
    def create_point_train_generator_DA(points, folder, batch_size, input_size, output_size, upscale=False):
        feature_count = len(points)
        feature_counter = 0
        while True:
            source_images = []
            target_images = []
            source_top = []
            source_left = []
            target_top = []
            target_left = []
            source_imgs = []
            target_imgs = []

            sampled_source = [] # to store the values of DA
            DA = []
            while len(source_images) < batch_size:
                if feature_counter == (feature_count-1):
                    feature_counter = 0

                point = points[feature_counter]
                sampled_source.append(point[3])
                # if int(point[3])==1: # to select the points with DA equals 1
                #     sampled_source.append(feature_counter)
                feature_counter += 1

                # source_patch, target_patch = DataManager.crop_by_point(point, folder, input_size, output_size, upscale=upscale)

                # print(np.amin(source_patch), np.amax(source_patch), np.amin(target_patch), np.amax(target_patch))     
                # cv2.imshow("source", source_patch)
                # cv2.imshow("target", target_patch)
                # cv2.waitKey()
                

                """
                print(np.amin(source_patch), np.amax(source_patch))
                print(np.amin(target_patch), np.amax(target_patch))
                """

                """
                u = str(uuid.uuid4())
                source_patch_shifted = source_patch[:, :, [2,1,0]]
                cv2.imwrite("Z:/_Sandbox/train_dump/" + u + "source.png", source_patch_shifted * 255)
                cv2.imwrite("Z:/_Sandbox/train_dump/" + u + "target.png", target_patch * 255)
                """
                touple_source,touple_target = DataManager.crop_by_point_new(point, folder, input_size, output_size, upscale=upscale)

                source_imgs.append(touple_source[0])
                target_imgs.append(touple_target[0])
                source_images.append(touple_source[1])
                target_images.append(touple_target[1])
                source_top.append(touple_source[3])
                source_left.append(touple_source[2])
                target_left.append(touple_target[2])
                target_top.append(touple_target[3])

            ####Conduct data autmentation
            for i in range(0,len(sampled_source)):
                idx = sampled_source[i]
                if int(idx)==1: # the points to be augmented
                    source_arr = source_imgs[i]
                    target_arr = target_imgs[i]
                    topS = source_top[i]
                    leftS = source_left[i]
                    topT = target_top[i]
                    leftT = target_left[i]
                    input_patch,arr_source =  DataAugmentation.data_aug_init(source_arr,topS,leftS,topT,leftT)
                    img_aug_source,img_aug_target = DataAugmentation.data_aug_random(input_patch,arr_source,target_arr)
                    source_images[i] = img_aug_source
                    target_images[i] = img_aug_target

            sources_array = np.asarray(source_images)
            targets_array = np.asarray(target_images)

            yield(sources_array, targets_array)
            # return(sources_array, targets_array)




    @staticmethod
    def create_point_predict_generator(points, folder, batch_size, input_size, output_size, upscale=False):
        feature_count = len(points)

        feature_counter = 0
        while True:
            source_images = []
            target_images = []

            while len(source_images) < batch_size:
                if feature_counter == (feature_count-1):
                    feature_counter = 0

                point = points[feature_counter]
                feature_counter += 1

                source_patch, target_patch = DataManager.crop_by_point(point, folder, input_size, output_size, upscale=upscale)

                source_images.append(source_patch)
                target_images.append(target_patch)

            sources_array = np.asarray(source_images)
            targets_array = np.asarray(target_images)

            yield(sources_array, targets_array)