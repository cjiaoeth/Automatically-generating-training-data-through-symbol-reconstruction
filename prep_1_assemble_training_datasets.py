#combine road data into one dataset

import fiona, os, json
from shapely.geometry import mapping, shape, LineString, Polygon
from fiona.crs import from_epsg

output_dir_roads = "training/roads/"



scale_dir = "../../../corrected/25"
year_dirs = os.listdir(scale_dir)
for year_dir in year_dirs:
    
    target_dir_roads    = output_dir_roads    + year_dir
    
    roads_files = []
    
    if not os.path.exists(target_dir_roads):
        os.makedirs(target_dir_roads)
    
    sheet_dir = scale_dir + "/" + year_dir
    
    roads_file = sheet_dir + "/roads.shp"
    if os.path.isfile(roads_file):
        roads_files.append(roads_file)

    
       
    
    schema = {
        'geometry': "LineString",
        'properties': {},
    }
    
    out_shp = target_dir_roads + "/road_merge.shp"
    with fiona.open(out_shp, 'w', 'ESRI Shapefile', schema, crs=from_epsg(21781)) as c:
        for roads_file in roads_files:
            print(roads_file)
            features = fiona.open(roads_file)
            
            for feature in features:
                if feature["geometry"] is None:
                    print("WARNING: Feature is empty.")
                    continue
                
                
                feature["properties"].clear()
                # feature["geometry"] = mapping(shape(feature["geometry"]).buffer(0))
                    
                c.write(feature)
