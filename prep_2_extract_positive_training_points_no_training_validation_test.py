import fiona, os, json
from shapely.geometry import mapping, shape, LineString, Polygon, Point
from osgeo import ogr, gdal
import numpy as np
from skimage.morphology import medial_axis, skeletonize


num_roads_points = 0


schema = {
    'geometry': "Point",
    'properties': {},
}

resolution = 1.25
sampling_fraction = 50 #used to show the progress and points selection
min_samples = 3


roads_dir = "training/roads"
    
year_dirs = os.listdir(roads_dir)
for year_dir in year_dirs:

    out_shp = roads_dir + "/" + year_dir + "/points.shp"

    with fiona.open(out_shp, 'w', 'ESRI Shapefile', schema) as c:
    
        file_location = roads_dir + "/" + year_dir + "/road_merge.shp"    
        driver = ogr.GetDriverByName('ESRI Shapefile')
        datasource = driver.Open(file_location, 0)
        layer = datasource.GetLayer()
        
        featureCount = layer.GetFeatureCount()
        
        # iterate over single features
        for i in range(featureCount):
            if (i % 100 == 0):
                print(str(i) + "/" + str(featureCount))
            layer.SetAttributeFilter("FID IN (" + str(i) + ")")
            
            feature = layer.GetNextFeature()
            geom = feature.GetGeometryRef()
            x_min, x_max, y_min, y_max = geom.GetEnvelope()
            
            #print(x_min, x_max, y_min, y_max)
            
            width = round((x_max - x_min) / resolution)
            height = round((y_max - y_min) / resolution)
            
            ######
            raster_name = "/vsimem/" + year_dir + "_" + str(i) + ".tif"
            
            if width<1 and height<1:
                target_ds = gdal.GetDriverByName("GTiff").Create(raster_name, 1, 1, 1, gdal.GDT_Byte, options=["COMPRESS=PACKBITS"])
            if width<1 and height>=1:
                target_ds = gdal.GetDriverByName("GTiff").Create(raster_name, 1, height, 1, gdal.GDT_Byte, options=["COMPRESS=PACKBITS"])
            if height<1 and width>=1:
                target_ds = gdal.GetDriverByName("GTiff").Create(raster_name, width, 1, 1, gdal.GDT_Byte, options=["COMPRESS=PACKBITS"])
            if width>=1 and height>=1:    
                target_ds = gdal.GetDriverByName("GTiff").Create(raster_name, width, height, 1, gdal.GDT_Byte, options=["COMPRESS=PACKBITS"])


            # target_ds.SetGeoTransform((x_min, resolution, 0, y_max, 0, -resolution))
            target_ds.SetGeoTransform((x_min, resolution, 0, y_max, 0, -resolution))

            
            band = target_ds.GetRasterBand(1)
            NoData_value = -999999
            band.SetNoDataValue(NoData_value)
            band.FlushCache()

            gdal.RasterizeLayer(target_ds, [1], layer)
            ##the matrix of image
            #normalization
            arr = (np.array(target_ds.GetRasterBand(1).ReadAsArray()) / 255).astype(np.uint8)
            # skel, _ = medial_axis(arr, return_distance=True)
            #print(arr.shape, np.amax(arr), np.amin(arr), np.sum(arr), arr.shape[0] * arr.shape[1])
            #acquire coordinates of white points(road sgment)
            feature_coordinates = np.argwhere(arr == 1)
            #determine the number of points, depends on sampling fraction
            num_random_points = max(int(np.sum(arr) / sampling_fraction), min_samples)
            #shuffle the indices
            indices = np.arange(np.sum(arr))
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
            
            #print(feature_geocoordinates)
            
            
            for p_coords in feature_geocoordinates:
                p = Point(p_coords[0], p_coords[1])
                feature = {"properties" : {}, "geometry": mapping(p)}
                c.write(feature)
                num_roads_points += 1
            
            
            layer.ResetReading()
            gdal.Unlink(raster_name)

print("roads points: ", num_roads_points)  
            
            