from osgeo import ogr, gdal
import osgeo, os


import utils
from segmentation_manager import SegmentationManager
from data_manager import DataManager


sm = SegmentationManager("F:/extraction/25/20201106/instances/", "F:/DL/models/")
#sm.create_instance("instance_deepwater_siegfried-small-batch-multi-4", "U-Net-Small-Batch-Multi", {"num_models" : 10, "num_base_filters" : 16}, ["roads"], True)

dummy_instance = sm.get_instance("instance_deepwater_siegfried-small-batch-multi-4")

"""
training_points = utils.NDFileManager.file("F:/extraction/25/20201106/training_points_aggregated.nd")
vector_data = utils.NDFileManager.file("F:/extraction/25/20201106/training_datasets.nd")
sheet_data = utils.NDFileManager.file("F:/extraction/25/20201106/training_sheets.nd")

batch_size = 16

training_data_location = dummy_instance.prepare_training_data(training_points, sheet_data, vector_data)

#dummy_instance.train(training_data_location, epochs=100, early_stopping_patience=50, batch_size=batch_size, upscale=False)
dummy_instance.train(training_data_location, epochs=200, early_stopping_patience=50, batch_size=batch_size, upscale=False)
"""

"""
testing_data_location = dummy_instance.prepare_testing_data(training_points, sheet_data, vector_data)
dummy_instance.test(testing_data_location, upscale=False, batch_size=batch_size)
"""

"""
prediction_data_location = dummy_instance.prepare_prediction_data(training_points, sheet_data, vector_data)
dummy_instance.predict(predicteion_data_location, upscale=False, batch_size=batch_size)
"""


target_path = "F:/predictions/25/1940/"
base_path = "F:/sheets/25/1940/"
#sheets = ["rgb_TA_017_1940.tif", "rgb_TA_018_1940.tif", "rgb_TA_019_1940.tif"]
sheets = ["rgb_TA_017_1940.tif"]


for sheet in sheets:
    sheet_path = base_path + sheet
    prediction_path = target_path + sheet

    dummy_instance.predict_sheet(sheet_path, prediction_path, batch_size = 20, resolution = 1.25, padding = 32, img_size=64, upsample=False)





