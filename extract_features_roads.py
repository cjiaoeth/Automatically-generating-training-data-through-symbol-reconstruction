import cv2, gdal
from osgeo import osr
import numpy as np
import networkx as nx
import fiona, itertools
from rdp import rdp
import ogr
import math
import os


from shapely.geometry import LineString, MultiPoint, MultiLineString, Point, MultiPolygon, Polygon, mapping, shape

from skimage.morphology import medial_axis, skeletonize

from shapely.ops import snap, unary_union, linemerge



# converts a binary image to a network (white pixels are the nodes)
def imageToNetwork(img, moor_neighborhood = False):

    indices = np.where(img == 1)
    indexTuples = list(zip(*indices))

    G=nx.Graph()
    G.add_nodes_from(indexTuples)

    
    d = 2
    counter = 0
    for iT in indexTuples:
       if (counter % 100 == 0):
          #print("Adding index tuple " + str(counter) + " of " + str(len(indexTuples)))
          pass
          
       counter += 1
       
       riT = (iT[0], iT[1] + 1)
       biT = (iT[0] + 1, iT[1])
       
       if riT[1] < img.shape[1] and img[riT[0], riT[1]] == 1:
           G.add_edge(iT, riT)
       
       if biT[0] < img.shape[0] and img[biT[0], biT[1]] == 1:
           G.add_edge(iT, biT)
       
       if (moor_neighborhood):
          rbiT = (iT[0] + 1, iT[1] + 1)
          lbiT = (iT[0] - 1, iT[1] + 1)
          if rbiT[1] < img.shape[1] and rbiT[0] < img.shape[0] and img[rbiT[0], rbiT[1]] == 1:
             G.add_edge(iT, rbiT)
         
          if lbiT[1] < img.shape[1] and lbiT[0] > -1 and img[lbiT[0], lbiT[1]] == 1:
             G.add_edge(iT, lbiT)
    
    
    return G



def write_raster(path, sheet, reference_ds):
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(path, sheet.shape[1], sheet.shape[0], 1,
                            gdal.GDT_Byte, ['COMPRESS=LZW'])
                            
    outdata.SetGeoTransform(reference_ds.GetGeoTransform())

    outdata.SetProjection(reference_ds.GetProjection())
    # outdata.GetRasterBand(1).SetNoDataValue(nodata)
    outdata.GetRasterBand(1).WriteArray(np.squeeze(sheet))
    outdata.FlushCache()
    outdata = None
    band = None

    ds = None

    
def remove_small_features(orig_img, bin_img, threshold, invert=False):
    components = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    num_components = components[0]

    for i in range(0, num_components):
        #print(str(i) + "/" + str(num_components))
        
        stats = components[2][i]
        start_x = stats[0]
        start_y = stats[1]
        width   = stats[2]
        height  = stats[3]

        cropped_labels = components[1][start_y:start_y+height, start_x:start_x+width]
        cropped_orig = bin_img[start_y:start_y+height, start_x:start_x+width]
        component = (cropped_labels == i)
        
        if (not np.sum(cropped_orig[component]) == 0) and np.sum(component) < threshold:
            if invert:
                orig_img[start_y:start_y+height, start_x:start_x+width][component] = 1
            else:
                orig_img[start_y:start_y+height, start_x:start_x+width][component] = 0


                
                
def compute_degrees(img):
    degree_raster = np.zeros(img.shape)
    skeleton_streams_reduced = img[1:-1, 1:-1].copy()

    left_pixels      = img[0:-2 ,1:-1]
    right_pixels     = img[2:   ,1:-1]
    top_pixels       = img[1:-1 ,0:-2]
    bottom_pixels    = img[1:-1 ,2:  ]
        
    top_left_pixels      = img[0:-2 ,0:-2]
    top_right_pixels     = img[2:   ,0:-2]
    bottom_left_pixels   = img[0:-2 ,2:  ]
    bottom_right_pixels  = img[2:   ,2:  ]
        
    """
    print(left_pixels.shape)
    print(degree_raster.shape)
    print(skeleton_streams_reduced.shape)
    """

    degree_raster[1:-1, 1:-1] += skeleton_streams_reduced & left_pixels
    degree_raster[1:-1, 1:-1] += skeleton_streams_reduced & right_pixels
    degree_raster[1:-1, 1:-1] += skeleton_streams_reduced & top_pixels
    degree_raster[1:-1, 1:-1] += skeleton_streams_reduced & bottom_pixels

    degree_raster[1:-1, 1:-1] += skeleton_streams_reduced & top_left_pixels
    degree_raster[1:-1, 1:-1] += skeleton_streams_reduced & top_right_pixels
    degree_raster[1:-1, 1:-1] += skeleton_streams_reduced & bottom_left_pixels
    degree_raster[1:-1, 1:-1] += skeleton_streams_reduced & bottom_right_pixels
    
    
    return degree_raster
    

def morphological_transformation(img, kernel):
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return opening
    
                
      
target_path = "F:/predictions/25/1940/Josianne/"
base_path = "F:/sheets/25/Josianne/"
#sheets = ["rgb_TA_017_1940.tif", "rgb_TA_018_1940.tif", "rgb_TA_019_1940.tif"]
# files = ["rgb_TA_017_1940.tif", "rgb_TA_018_1940.tif", "rgb_TA_021_bis_1940.tif", "rgb_TA_021_1940.tif", "rgb_TA_160_1940.tif", "rgb_TA_161_1940.tif"]
files = ["rgb_TA_021_bis_1940.tif"]
# files = ["rgb_TA_017_1940.tif", "rgb_TA_018_1940.tif"]


counter = 0
threshold = 0.5
for file_raw in files:
    name = file_raw.rstrip()
    counter += 1
    print(str(counter) + " " + name)
    
    input_path = target_path + file_raw
    temp_path = target_path + file_raw[:-4]
    
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    ds = gdal.Open(input_path)

    # roads_arr   = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(np.uint8)
    roads_array   = np.array(ds.GetRasterBand(1).ReadAsArray())
    roads_arr_copy = np.where(roads_array<threshold,0,1) #set values according to the conditions
    roads_arr = roads_arr_copy.astype(np.uint8)
    roads_patch_threshold = 100
    remove_small_features(roads_arr, roads_arr, roads_patch_threshold)
    
    
    skeleton_path = temp_path + "/skeleton.tif"
    roads_path = temp_path + "/roads.shp"
    generalized_roads_path = temp_path + "/generalized_roads.shp"
    
    
    print("Creating roads")
    skeleton_roads, distance = medial_axis(roads_arr, return_distance=True)
    write_raster(skeleton_path, skeleton_roads, ds)

    degree_raster = compute_degrees(skeleton_roads)
    path_raster = degree_raster.copy()
    path_raster[path_raster > 0] = 1

    components = cv2.connectedComponentsWithStats(path_raster.astype("uint8"), connectivity=8)
    num_components = components[0]

    long_paths = []
    gt = ds.GetGeoTransform()
    road_threshold = 5
    for i in range(0, num_components):
        print(str(i) + "/" + str(num_components))
        stats = components[2][i]
        start_x = stats[0]
        start_y = stats[1]
        width   = stats[2]
        height  = stats[3]

        cropped_labels = components[1][start_y:start_y+height, start_x:start_x+width]
        cropped_distances = distance[start_y:start_y+height, start_x:start_x+width]
        
        component = (cropped_labels == i)

        if np.sum(cropped_labels[component]) == 0:
            continue
        
        if np.sum(component) == 1:
            continue
        
        G = imageToNetwork(component, True)
        comp_paths = []
        
        chains = nx.chain_decomposition(G)
        
        for chain in chains:
            nodes = []
            is_first = True
            for edge in chain:
                if is_first:
                    nodes.append(edge[0])
                    nodes.append(edge[1])
                    is_first = False
                    
                else:
                    if edge[0] == nodes[-1]:
                        nodes.append(edge[1])
                    else:
                        nodes.append(edge[0])
        
            cycle_threshold = 10    
            path_adjusted = [((y + start_x) * gt[1] + gt[0] + 0.5 * gt[1], (x + start_y) * gt[5] + gt[3] + 0.5 * gt[5]) for x, y in nodes]
            cycle_ls = LineString(path_adjusted)
            
            if cycle_ls.length > cycle_threshold:
                comp_paths.append(cycle_ls)
        

        
        """
        remove = [node for node,degree in dict(G.degree()).items() if degree > 2]
        G.remove_nodes_from(remove)
        """
        if len(G.nodes()) > 0:
            nodes = [node for node,degree in dict(G.degree()).items() if degree == 1]
            if len(nodes) < 2:
                continue
            
            start_node = nodes[0]
            other_nodes = nodes[1:]
            
            for end_node in other_nodes:
                path = nx.shortest_path(G, start_node, end_node)
            
                path_adjusted = [((y + start_x) * gt[1] + gt[0] + 0.5 * gt[1], (x + start_y) * gt[5] + gt[3] + 0.5 * gt[5]) for x, y in path]
                comp_paths.append(LineString(path_adjusted))
        
            component_paths = []
            union = unary_union(comp_paths)
            if union.geom_type == "LineString":
               component_paths.append(union)
            else:
                multi_lines = linemerge(union)
                if multi_lines.geom_type == "MultiLineString":            
                    for line_string in multi_lines:
                        component_paths.append(line_string)
                else:
                   component_paths.append(multi_lines)
        
        
            all_paths = MultiLineString(component_paths)
            for ls in component_paths:
                if ls.length < road_threshold:
                    subtracted_paths = all_paths.difference(ls)
                    intersection = subtracted_paths.intersection(ls)
                    if intersection.is_empty:
                        continue
                    
                    # should only be two points, which means that it is an intermediate segment (no dangle)
                    if intersection.geom_type == "MultiPoint": 
                        long_paths.append(ls)
                        
                else:
                    long_paths.append(ls)
                
            
        
        
    filtered_linestrings = [linestring for linestring in long_paths if linestring.length > road_threshold]


    schema = {
        'geometry': "LineString",
        'properties': {},
    }


    with fiona.open(roads_path, 'w', 'ESRI Shapefile', schema) as c:
        for filtered_path in filtered_linestrings:
            c.write({"properties":{}, "geometry": mapping(LineString(filtered_path))})
            
    e = 1.5
    generalized_linestrings = [LineString(filtered_path.simplify(e)) for filtered_path in filtered_linestrings]

    schema = {
        'geometry': "LineString",
        'properties': {},
    }

    with fiona.open(generalized_roads_path, 'w', 'ESRI Shapefile', schema) as c:
        for generalized_linestring in generalized_linestrings:
            c.write({"properties":{}, "geometry": mapping(generalized_linestring)})    

            
            
            
            
            
            
            
            
            
        
        


