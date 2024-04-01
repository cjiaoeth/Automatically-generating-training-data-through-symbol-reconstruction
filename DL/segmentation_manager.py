import os, json, shutil
from osgeo import ogr
from pathlib import Path
from utils import raise_error, get_timestamp, ndshp2dict, trim_GeoJSON, sort_by_key
import random
import gdal
import uuid
import cv2
import random

from model_manager import model_manager
from data_manager import DataManager

from keras.callbacks import CSVLogger, EarlyStopping


import numpy as np

class SegmentationModel:
    def __init__(self):
        pass

    def get_input_size(self):
        return (320, 320, 4)

    def get_output_size(self):
        return (200, 200, 1)




class SegmentationInstance:
    def __init__(self, location):
        self.location = location
        self.load()

        self.model = model_manager.create_model(self.metadata["model_name"])

        print(self.metadata)
        if self.metadata["trained"]:
            print("loading")
            self.model.load(str(self.location / "model"))

    @staticmethod
    def create_point_layer(path):
        driver = ogr.GetDriverByName('Esri Shapefile')
        ds = driver.CreateDataSource(str(path))
        layer = ds.CreateLayer('', None, ogr.wkbPoint)
        layer.CreateField(ogr.FieldDefn('scale', ogr.OFTString))
        layer.CreateField(ogr.FieldDefn('year', ogr.OFTString))
        ds = layer = None


    @staticmethod
    def create(instance_path, instance_name, model_name, parameters, layer_names, overwrite = False):

        creation_path = instance_path / instance_name

        # set up instance structure
        if not os.path.exists(creation_path) or overwrite:
            os.makedirs(creation_path / "training", exist_ok=overwrite)
            os.makedirs(creation_path / "prediction", exist_ok=overwrite)
            os.makedirs(creation_path / "testing", exist_ok=overwrite)
            os.makedirs(creation_path / "model", exist_ok=overwrite)
            os.makedirs(creation_path / "temp", exist_ok=overwrite)

            with open(creation_path / "metadata.json", 'w') as outfile:
                params = dict(parameters)
                params["num_classes"] = len(layer_names)
                json.dump({"model_name": model_name, "instance_name" : instance_name, "parameters" : params, "layer_names" : layer_names, "trained" : False, "training_datasets" : []}, outfile, indent=4)

        else:
            raise_error("Could not create instance directories for " + str(creation_path) + " .")

        return SegmentationInstance(creation_path)


    def prepare_training_data(self, shp_vector_layer, sheet_data, vector_data):
        points = ndshp2dict(shp_vector_layer, lambda x: x.GetField('type') == "validation" or x.GetField('type') == "training")
        padding = max(self.model.get_extents()["input"] / 2, self.model.get_extents()["output"] / 2)

        u = get_timestamp()
        file_name = os.path.basename(shp_vector_layer.get_location())[:-4]
        target_path = str(self.location / "training" / file_name) + "_" + u
        os.makedirs(target_path)

        dm = DataManager(sheet_data, vector_data)
        dm.extract_region(points, target_path, self.metadata["layer_names"], padding)
        self.metadata["training_datasets"].append(shp_vector_layer.get_location())

        return target_path


    def clear_training_data(self):
        # not implemented for now
        pass


    def train_DA(self, data_location, batch_size = 20, epochs = 30, early_stopping_patience = 20, early_stopping_monitor = "val_binary_accuracy", upscale=False):

        points = trim_GeoJSON(data_location + "/points.geojson")

        training_points   = [p for p in points if p[2] == "training"]
        validation_points = [p for p in points if p[2] == "validation"]
        random.shuffle(training_points)
        random.shuffle(validation_points)

        self.model.create(self.metadata["parameters"])

        training_generators = []
        validation_generators = []
        callbacks_list = []
        for i in range(self.metadata["parameters"]["num_models"]):
            training_generator = DataManager.create_point_train_generator_DA(training_points, data_location, 
                                                                          batch_size, 
                                                                          self.model.get_extents()["input"], 
                                                                          self.model.get_extents()["output"], upscale=upscale)
            training_generators.append(training_generator)

            validation_generator = DataManager.create_point_train_generator_DA(validation_points, data_location,
                                                                            batch_size,
                                                                            self.model.get_extents()["input"],
                                                                            self.model.get_extents()["output"], upscale=upscale)

            validation_generators.append(validation_generator)


            csv_logger = CSVLogger(str(self.location) + "/model/" + str(i) + "training.log")
            early_stopper = EarlyStopping(monitor=early_stopping_monitor, min_delta=0, patience=early_stopping_patience, verbose=0, mode='auto',
                                          baseline=None, restore_best_weights=True)
            callbacks_list.append([csv_logger, early_stopper])

        self.model.compile()
        self.model.train(training_generators, validation_generators, len(training_points), len(validation_points), batch_size, epochs, callbacks_list = callbacks_list, location = self.location / "model/")

        self.metadata["epchos"] = epochs
        self.metadata["early_stopping_patience"] = early_stopping_patience
        self.metadata["early_stopping_monitor"] = early_stopping_monitor
        self.metadata["trained"] = True

        self.save()
        #self.model.save(self.location / "model/")
        
    def train(self, data_location, batch_size = 20, epochs = 30, early_stopping_patience = 20, early_stopping_monitor = "val_binary_accuracy", upscale=False):

        points = trim_GeoJSON(data_location + "/points.geojson")

        training_points   = [p for p in points if p[2] == "training"]
        validation_points = [p for p in points if p[2] == "validation"]

        self.model.create(self.metadata["parameters"])

        training_generators = []
        validation_generators = []
        callbacks_list = []
        for i in range(self.metadata["parameters"]["num_models"]):
            training_generator = DataManager.create_point_train_generator(training_points, data_location, 
                                                                          batch_size, 
                                                                          self.model.get_extents()["input"], 
                                                                          self.model.get_extents()["output"], upscale=upscale)
            training_generators.append(training_generator)

            validation_generator = DataManager.create_point_train_generator(validation_points, data_location,
                                                                            batch_size,
                                                                            self.model.get_extents()["input"],
                                                                            self.model.get_extents()["output"], upscale=upscale)

            validation_generators.append(validation_generator)


            csv_logger = CSVLogger(str(self.location) + "/model/" + str(i) + "training.log")
            early_stopper = EarlyStopping(monitor=early_stopping_monitor, min_delta=0, patience=early_stopping_patience, verbose=0, mode='auto',
                                          baseline=None, restore_best_weights=True)
            callbacks_list.append([csv_logger, early_stopper])

        self.model.compile()
        self.model.train(training_generators, validation_generators, len(training_points), len(validation_points), batch_size, epochs, callbacks_list = callbacks_list, location = self.location / "model/")

        self.metadata["epchos"] = epochs
        self.metadata["early_stopping_patience"] = early_stopping_patience
        self.metadata["early_stopping_monitor"] = early_stopping_monitor
        self.metadata["trained"] = True

        self.save()
        #self.model.save(self.location / "model/")

    def prepare_testing_data(self, shp_vector_layer, sheet_data, vector_data):
        points = ndshp2dict(shp_vector_layer, lambda x: x.GetField('type') == "testing")
        padding = max(self.model.get_extents()["input"] / 2, self.model.get_extents()["output"] / 2)

        u = get_timestamp()
        file_name = os.path.basename(shp_vector_layer.get_location())[:-4]
        target_path = str(self.location / "testing" / file_name) + "_" + u
        os.makedirs(target_path)

        dm = DataManager(sheet_data, vector_data)
        dm.extract_region(points, target_path, self.metadata["layer_names"], padding)

        return target_path




    def test(self, data_location, batch_size = 10, upscale=False):

        points = trim_GeoJSON(data_location + "/points.geojson")
        points_sorted = sort_by_key(points, 3, False)

        in_size = self.model.get_extents()["input"]
        out_size = self.model.get_extents()["output"]

        for key, pnts in points_sorted.items():
            key_path = data_location + "/" + key

            in_paths = []
            out_paths = []
            m_paths = []
            source_images = []
            target_images = []
            points_cache = []
            point_counter = 0

            pnts_length = len(pnts)
            for a in range(pnts_length):
                testing_point = pnts[a]
                source_patch, target_patch = DataManager.crop_by_point(testing_point, data_location, in_size, out_size, upscale=upscale)

                resolution = DataManager.get_resolution(testing_point[3])
                if upscale:
                    resolution = resolution / 2

                source_images.append(source_patch)
                target_images.append(target_patch)
                points_cache.append(testing_point)

                if a == pnts_length - 1:
                    is_last_batch = True
                else:
                    is_last_batch = False


                if len(source_images) == batch_size or is_last_batch:
                    np_source_images = np.asarray(source_images)

                    if is_last_batch:
                        num_patches = (a % batch_size) + 1
                    else:
                        num_patches = batch_size

                    predictions = self.model.predict(np_source_images, num_patches)

                    for i in range(num_patches):
                        prediction = predictions[i]
                        target = target_images[i]
                        point = points_cache[i]

                        if prediction.shape[2] == 1:
                            prediction = prediction[:, :, 0]

                        if target.shape[2] == 1:
                            target = target[:, :, 0]

                        # out point
                        out_driver = gdal.GetDriverByName("GTiff")
                        out_path = key_path + "/" + str(point_counter) + "_prediction.tif"
                        out_paths.append(out_path)
                        out_data = out_driver.Create(out_path, out_size, out_size, 1, gdal.GDT_Byte, ['COMPRESS=LZW'])

                        out_geotransform = [point[0] - 0.5 * out_size * resolution, resolution, 0,
                                            point[1] + 0.5 * out_size * resolution, 0, -resolution]

                        out_data.SetGeoTransform(out_geotransform)

                        out_data.GetRasterBand(1).WriteArray(prediction)
                        out_data.FlushCache()
                        out_data = None


                        # in point
                        in_driver = gdal.GetDriverByName("GTiff")
                        in_path = key_path + "/" + str(point_counter) + "_training.tif"
                        in_paths.append(in_path)
                        in_data = in_driver.Create(in_path, out_size, out_size, 1, gdal.GDT_Byte, ['COMPRESS=LZW'])

                        in_geotransform = [point[0] - 0.5 * out_size * resolution, resolution, 0,
                                           point[1] + 0.5 * out_size * resolution, 0, -resolution]

                        in_data.SetGeoTransform(in_geotransform)

                        in_data.GetRasterBand(1).WriteArray(target)
                        in_data.FlushCache()
                        in_data = None


                        # statistics
                        target_discrete = (target > 0.5)
                        prediction_discrete = (prediction > 0.5)

                        tp =  prediction_discrete &  target_discrete
                        tn = ~prediction_discrete & ~target_discrete
                        fp = ~prediction_discrete &  target_discrete
                        fn =  prediction_discrete & ~target_discrete


                        m_driver = gdal.GetDriverByName("GTiff")
                        m_path = key_path + "/" + str(point_counter) + "_metrics.tif"
                        m_paths.append(m_path)
                        m_data = in_driver.Create(m_path, out_size, out_size, 1, gdal.GDT_Byte, ['COMPRESS=LZW'])

                        m_geotransform = [point[0] - 0.5 * out_size * resolution, resolution, 0,
                                          point[1] + 0.5 * out_size * resolution, 0, -resolution]

                        m_data.SetGeoTransform(m_geotransform)

                        metrics_array = np.empty(target.shape, dtype=np.byte)
                        metrics_array[tp] = 1
                        metrics_array[tn] = 2
                        metrics_array[fp] = 3
                        metrics_array[fn] = 4

                        m_data.GetRasterBand(1).WriteArray(metrics_array)
                        m_data.FlushCache()
                        m_data = None

                        point_counter += 1


                    points_cache = []
                    source_images = []
                    target_images = []
                    np_source_images = None


            vrt_path = key_path + "/" + "prediction.vrt"
            vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', addAlpha=True)
            gdal.BuildVRT(vrt_path, out_paths, options=vrt_options)


            vrt_path = key_path + "/" + "training.vrt"
            vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', addAlpha=True)
            gdal.BuildVRT(vrt_path, in_paths, options=vrt_options)


            vrt_path = key_path + "/" + "metrics.vrt"
            vrt_options = gdal.BuildVRTOptions(resampleAlg='linear', addAlpha=True)
            gdal.BuildVRT(vrt_path, m_paths, options=vrt_options)




    def prepare_prediction_data(self, shp_vector_layer, sheet_data, vector_data):
        points = ndshp2dict(shp_vector_layer, lambda x: x.GetField('type') == "prediction")
        padding = max(self.model.get_extents()["input"] / 2, self.model.get_extents()["output"] / 2)

        u = get_timestamp()
        file_name = os.path.basename(shp_vector_layer.get_location())[:-4]
        target_path = str(self.location / "prediction" / file_name) + "_" + u
        os.makedirs(target_path)

        dm = DataManager(sheet_data, vector_data)
        dm.extract_region(points, target_path, self.metadata["layer_name"], padding, extract_labels=False)

        return target_path



    def predict(self, data_location, batch_size = 10, upscale=False):
        points = trim_GeoJSON(data_location + "/points.geojson")
        points_sorted = sort_by_key(points, 3, False)

        in_size = self.model.get_extents()["input"]
        out_size = self.model.get_extents()["output"]

        for key, pnts in points_sorted.items():
            key_path = data_location + "/" + key

            out_paths = []
            source_images = []
            points_cache = []
            point_counter = 0

            pnts_length = len(pnts)
            for a in range(pnts_length):
                prediction_point = pnts[a]
                source_patch = DataManager.crop_sheet_by_point(prediction_point, data_location, in_size, upscale=upscale)

                resolution = DataManager.get_resolution(prediction_point[3])

                source_images.append(source_patch)
                points_cache.append(prediction_point)

                if a == pnts_length - 1:
                    is_last_batch = True
                else:
                    is_last_batch = False


                if len(source_images) == batch_size or is_last_batch:
                    np_source_images = np.asarray(source_images)

                    if is_last_batch:
                        num_patches = (a % batch_size) + 1
                    else:
                        num_patches = batch_size

                    predictions = self.model.predict(np_source_images, num_patches)

                    for i in range(num_patches):
                        prediction = predictions[i]
                        point = points_cache[i]

                        if prediction.shape[2] == 1:
                            prediction = prediction[:, :, 0]

                        # out point
                        out_driver = gdal.GetDriverByName("GTiff")
                        out_path = key_path + "/" + str(point_counter) + "_prediction.tif"
                        out_paths.append(out_path)
                        out_data = out_driver.Create(out_path, out_size, out_size, 1, gdal.GDT_Byte, ['COMPRESS=LZW'])

                        out_geotransform = [point[0] - 0.5 * out_size * resolution, resolution, 0,
                                            point[1] + 0.5 * out_size * resolution, 0, -resolution]

                        out_data.SetGeoTransform(out_geotransform)

                        out_data.GetRasterBand(1).WriteArray(prediction)
                        out_data.FlushCache()
                        out_data = None

                        point_counter += 1


                    points_cache = []
                    source_images = []
                    target_images = []
                    np_source_images = None


            vrt_path = key_path + "/" + "prediction.vrt"
            vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', addAlpha=True)
            gdal.BuildVRT(vrt_path, out_paths, options=vrt_options)



    def predict_sheet(self, input_sheet_location, output_sheet_location, batch_size = 20, resolution = 1.25, padding = 60, img_size=200, upsample=False):
        if upsample:
            padding = int(padding / 2)
            img_size = int(img_size / 2)

        # retrieve geospatial information
        ds = gdal.Open(input_sheet_location)
        transform_in = ds.GetGeoTransform()
        if upsample:
            transform_out = (transform_in[0], transform_in[1] / 2, transform_in[2],
                             transform_in[3], transform_in[4], transform_in[5] / 2)
        else:
            transform_out = (transform_in[0], transform_in[1], transform_in[2], transform_in[3], transform_in[4], transform_in[5])


        bands = []
        for i in range(4):
            bands.append(ds.GetRasterBand(i+1).ReadAsArray())

        sheet = np.dstack(tuple(bands))



        
        # make sure that the sheet matches the model extent parameters
        excess_x = sheet.shape[1] % img_size
        excess_y = sheet.shape[0] % img_size
        
        if not excess_x == 0:
            additional_padding_x = img_size - excess_x
        else:
            additional_padding_x = 0
        
        if not excess_y == 0:
            additional_padding_y = img_size - excess_y
        else:
            additional_padding_y = 0


        if upsample:
            sheet_template = np.zeros((sheet.shape[0] * 2, sheet.shape[1] * 2, self.metadata["parameters"]["num_classes"]), np.float32)
        else:
            sheet_template = np.zeros((sheet.shape[0] + additional_padding_y, sheet.shape[1] + additional_padding_x, self.metadata["parameters"]["num_classes"]), np.float32)
            
            
        sheet_extended = np.zeros((int(sheet.shape[0] + 2*padding + additional_padding_y), int(sheet.shape[1] + 2*padding + additional_padding_x), 4), np.float32)
        sheet_extended[padding:-padding-additional_padding_y, padding:-padding-additional_padding_x,:] = sheet

        x_count = int((sheet_extended.shape[1] - padding * 2) / img_size)
        y_count = int((sheet_extended.shape[0] - padding * 2) / img_size)

        
        for y in range(y_count):
            print(str(y) + str("/") + str(y_count))
            for x in range(x_count):
                y_start = y * img_size
                y_end = y * img_size + img_size
                x_start = x * img_size
                x_end = x * img_size + img_size

                sub_img = sheet_extended[y_start:y_end + 2 * padding, x_start:x_end + 2 * padding] / 255
                if upsample:
                    sub_img = cv2.resize(sub_img, (sub_img.shape[1] * 2, sub_img.shape[0] * 2), interpolation = cv2.INTER_LINEAR)

                #print(np.amax(sub_img))
                sub_img_expanded = np.expand_dims(sub_img, axis=0)

                Y_pred = self.model.predict(sub_img_expanded, batch_size)[0]

                if upsample:
                    y_start_upsample = y * img_size * 2
                    y_end_upsample   = y * img_size * 2 + img_size * 2
                    x_start_upsample = x * img_size * 2
                    x_end_upsample   = x * img_size * 2 + img_size * 2
                    sheet_template[y_start_upsample:y_end_upsample, x_start_upsample:x_end_upsample, :] = Y_pred.copy()
                else:
                    sheet_template[y_start:y_end, x_start:x_end, :] = Y_pred.copy()



        """
        if extension_y == 0:
            extension_y = -sheet_pad_clean_cropped.shape[0]
        if extension_x == 0:
            extension_x = -sheet_pad_clean_cropped.shape[1]
        """
        
        sheet_out = sheet_template[0:sheet.shape[0], 0:sheet.shape[1], :]


        # write raster
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(output_sheet_location, sheet_out.shape[1], sheet_out.shape[0], self.metadata["parameters"]["num_classes"],
                                gdal.GDT_Float32, ['COMPRESS=LZW'])
        outdata.SetGeoTransform(transform_out)

        outdata.SetProjection(ds.GetProjection())

        # outdata.GetRasterBand(1).SetNoDataValue(nodata)
        bands = []
        for i in range(self.metadata["parameters"]["num_classes"]):
            outdata.GetRasterBand(i+1).WriteArray(np.squeeze(sheet_out[:, :, i]))

        outdata.FlushCache()
        outdata = None
        band = None

        ds = None



    def save(self):
        print("SAVING")
        with open(self.location / "metadata.json", 'w') as outfile:
            json.dump(self.metadata, outfile, indent=4)

    def load(self):
        with open(self.location / "metadata.json", "r") as metadata_location:
            self.metadata = json.load(metadata_location)


class SegmentationManager:
    def __init__(self, instance_path, model_path):
        self.instance_path = Path(instance_path)
        self.model_path = Path(model_path)

    def get_models(self):
        self.models = {}

    def get_instance_metadata(self):
        instance_files = os.listdir(self.instance_path)

        instance_dirs = {}

        for instance_file in instance_files:
            if os.path.isdir(self.instance_path / instance_file):
                with open(self.instance_path / instance_file / "metadata.json") as metadata_json:
                    metadata = json.load(metadata_json)

                    instance_dirs[instance_file] = metadata
        return instance_dirs



    def get_instance(self, instance_name):
        if instance_name in self.get_instance_metadata():
            return SegmentationInstance(self.instance_path / instance_name)
        else:
            raise_error("Could not get instance \"" + instance_name + "\": Instance does not exist.")


    def delete_instance(self, instance_name):
        if instance_name in self.get_instance_metadata():
            pass # not enabled right now for security reasons
            # shutil.rmtree(self.instance_path / instance_name)


        else:
            raise_error("Could not delete instance \"" + instance_name + "\": Instance does not exist.")



    def create_instance(self, instance_name, model, parameters, layer_names, overwrite = False):
        if not instance_name in self.get_instance_metadata() or overwrite:
            SegmentationInstance.create(self.instance_path, instance_name, model, parameters, layer_names, overwrite)
        else:
            raise_error("Instance name already in use.")








