from osgeo import ogr, gdal
import osgeo, os
from os import listdir
from os.path import isfile, join

import utils
from segmentation_manager import SegmentationManager
from data_manager import DataManager


sm = SegmentationManager("D:/Catherine/doctoral studies/symbolization reconstruction/training scenarios from the dell machine/20210831/instances/", "F:/DL/models/")
dummy_instance = sm.get_instance("instance_road_siegfried-small-batch-multi-4")


target_path = "D:/Catherine/doctoral studies/road extraction/test/old national maps_617 sheets/predictions/"
base_path = "D:/Catherine/doctoral studies/road extraction/test/old national maps_617 sheets/sheets/"
sheets = [f for f in listdir(base_path) if isfile(join(base_path, f))]
# sheets = ["LKg_1011_1971test.tif"]
# sheets = ["rgb_TA_072_1880.tif"]


for sheet in sheets:
    sheet_path = base_path + sheet
    prediction_path = target_path + sheet

    dummy_instance.predict_sheet(sheet_path, prediction_path, batch_size = 20, resolution = 1.25, padding = 32, img_size=64, upsample=False)





