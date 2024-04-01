from osgeo import ogr, gdal
import osgeo, os


import utils
from segmentation_manager import SegmentationManager
from data_manager import DataManager


# NEW 2 - WORKS VERY VERY WELL
sm = SegmentationManager("Z://extraction//50//1 1880//instances//", "D://GeoDL//DL//models//")
"""
sm.create_instance("instance_deepwater_siegfried-small-batch-multi-50-2", "U-Net-Small-Batch-Multi", {"num_models" : 10, "num_base_filters" : 16}, ["stream", "wetland", "riverlake"], True)
"""
dummy_instance = sm.get_instance("instance_deepwater_siegfried-small-batch-multi-50-2")

"""
training_points = utils.NDFileManager.file("Z:/extraction/50/1 1880/training_points_aggregated.nd")
vector_data = utils.NDFileManager.file("Z:/extraction/50/1 1880/training_datasets.nd")
sheet_data = utils.NDFileManager.file("Z:/extraction/50/1 1880/training_sheets.nd")

batch_size = 16


training_data_location = dummy_instance.prepare_training_data(training_points, sheet_data, vector_data)
dummy_instance.train(training_data_location, epochs=100, early_stopping_patience=50, batch_size=batch_size, upscale=False)
"""

"""
testing_data_location = dummy_instance.prepare_testing_data(training_points, sheet_data, vector_data)
dummy_instance.test(testing_data_location, upscale=False, batch_size=batch_size)
"""


"""
prediction_data_location = dummy_instance.prepare_prediction_data(training_points, sheet_data, vector_data)
dummy_instance.predict(predicteion_data_location, upscale=False, batch_size=batch_size)
"""

years = [1875, 1876, 1879, 1880, 1881, 1882, 1884, 1886, 1887, 1888, 1889, 1891, 1892, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1907, 1908, 1910, 1915, 1926]
#, , , ]
for year in years:
    base_dir = "Z://extraction//50//1 "+str(year)+"//"
    files = open(base_dir + "target_sheets.txt", "r")
    print(files)
    for file_raw in files:
        file = file_raw.rstrip()
        target_dir = base_dir + "//results_3//" + file

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        input_sheet_location = "Z://sheets//" + "rgb_" + file + ".tif"
        output_sheet_location = target_dir + "//" + file + "_predictions.tif"
        dummy_instance.predict_sheet(input_sheet_location, output_sheet_location, batch_size = 20, resolution = 2.50, padding = 32, img_size=64, upsample=False)






