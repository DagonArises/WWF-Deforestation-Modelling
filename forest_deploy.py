import numpy as np
from io import BytesIO, StringIO
from copy import copy
import xarray as xr
from google.cloud import storage
from onnxruntime import InferenceSession
import datetime
import cv2 as cv
from skimage.morphology import remove_small_objects

wcf_rf_model = 'wcfrf_eaAU.onnx'

# Function that loads the WCF RF model
# This is stored on GCS as a .onnx file
def get_model(model_name):
    client = storage.Client()
    bucket = client.get_bucket('terrakio-models-eu')
    blob = bucket.blob(model_name)
    model = BytesIO()
    blob.download_to_file(model)
    model.seek(0)
    
    return InferenceSession(model.read(), providers=["CPUExecutionProvider"])

# TODO: Discuss with team about ways of caching the model
model = get_model(wcf_rf_model)


forest_definitions = {
    'FAO': {
        'min_crown_cover': 0.1,   # minimum crown cover (as fraction)
        'min_cover_area': 0.5,    # minimum area (in ha)
        'min_height': 5           # minimum height (in m)
    },
    'Australian': {
        'min_crown_cover': 0.2,
        'min_cover_area': 0.2,
        'min_height': 2
    }
}

au_kernel_csv = """0,0,0,0.000527,0,0,0
0,0.0076873,0.039743,0.0498,0.039743,0.0076873,0
0,0.039743,0.05,0.05,0.05,0.039743,0
0.000527,0.0498,0.05,0.05,0.05,0.0498,0.000527
0,0.039743,0.05,0.05,0.05,0.039743,0
0,0.0076873,0.039743,0.0498,0.039743,0.0076873,0
0,0,0,0.000527,0,0,0"""

kernel_au = np.genfromtxt(StringIO(au_kernel_csv), delimiter=',').astype(np.float32)

fao_kernel_csv = """0,0,0.00085576,0.0070101,0.0095791,0.0070101,0.00085576,0,0
0,0.0039758,0.017857,0.02,0.02,0.02,0.017857,0.0039758,0
0.00085576,0.017857,0.02,0.02,0.02,0.02,0.02,0.017857,0.00085576
0.0070101,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.0070101
0.0095791,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.0095791
0.0070101,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.0070101
0.00085576,0.017857,0.02,0.02,0.02,0.02,0.02,0.017857,0.00085576
0,0.0039758,0.017857,0.02,0.02,0.02,0.017857,0.0039758,0
0,0,0.00085576,0.0070101,0.0095791,0.0070101,0.00085576,0,0"""

kernel_fao = np.genfromtxt(StringIO(fao_kernel_csv), delimiter=',').astype(np.float32)

forest_definitions['FAO']['kernel'] = kernel_fao
forest_definitions['Australian']['kernel'] = kernel_au


def map_forest(trees, min_crown_cover, min_cover_area, kernel, debug=False, **kwargs):
    x = cv.filter2D(trees.astype(np.float32), -1, kernel, borderType=cv.BORDER_CONSTANT) > min_crown_cover
    x = cv.filter2D(x.astype(np.float32), -1, kernel, borderType=cv.BORDER_CONSTANT) > (1 - min_crown_cover)
    x = remove_small_objects(x, min_size=int(100 * min_cover_area))  #trim forest
    x = remove_small_objects(~x, min_size=int(100 * min_cover_area))  # trim non-forest

    return ~x


# Take input ts: [2017, 2023]
# Output ds: ts1 = 2020 for ref_tree, ts2 to ts4 are 21-23 for now_tree
# Reference threshold is 0.04 based on calibration results
def allforests(da1, da2, da3, da4, da5, da6, da7, meta_theight, zhao_theight, definition='Australian', thresh = 0.04):

    # Step 1: Predict WCF from S2v2 reflectances
    arr = np.column_stack(tuple(da.values.flatten() for da in [da1,da2,da3,da4,da5,da6,da7]))
    output = model.run(None, {"X": arr.astype(np.float32)})[0]

    wcf = xr.DataArray(
        data=output.reshape(da1.shape),
        dims=['time', 'y', 'x'],
        coords={
            'time': da1.time,
            'y': da1.y,
            'x': da1.x
        }
    )

    # Remove water areas from WCF based on simple NDVI > 0.1 threshold
    # This is applied to each of the quarterly WCF values
    ndvi = (da4-da1) / (da4+da1)
    ndvimask = (ndvi > 0.1).values
    wcf.values = np.where(ndvimask, wcf.values, 0.0)


    # Step 2: Retain only annual minima from the quarterly WCF values
    # From quarterly WCF to annual WCF
    wcf = wcf.groupby('time.year').min(dim='time').rename({'year': 'time'})


    # Step 4: Calculate stats for reference period 
    wcf_ref = wcf.sel(time=wcf.time.isin([2017, 2018, 2019, 2020]))

    # Define NDVI range mask for reference period
    # take minimum and maximum ndvi for the 16 quarterly values in the 2017:2020 reference period
    # This mask filters minimum NDVI > 0.1 and difference between max and min NDVI < 0.5
    ndvi_ref = ndvi.sel(time=ndvi.time.isin([2017, 2018, 2019, 2020]))
    ndvimin = ndvi_ref.min(dim='time')
    ndvimax = ndvi_ref.max(dim='time')
    mask = (ndvimin > 0.1) * ( (ndvimax-ndvimin) < 0.5 )

    # Define trees as WCF > threshold at least 50% of the time over the reference period
    # Apply the NDVI range mask to the resulting trees
    trees_ref = (wcf_ref > thresh).mean('time').values >= 0.5
    trees_ref = np.where(mask.values, trees_ref, 0.0)

    # Calculate mean and standard deviation for reference period for Z-score calculation
    mean_ref = wcf_ref.mean('time').values
    std_ref = wcf_ref.std('time').values  

    
    # Step 5: Calculate Z-score for each year in the 2021-2023 WCF values
    forests_year = []
    input_years = [2021, 2022, 2023]
    for yr in input_years:
        wcf_now = wcf.sel(time=yr).values          # current last year
        z = (wcf_now - mean_ref) / std_ref         # calculate Z-score

        # Step 4: Crown cover mapping
        trees_gone = (((z < -3) & (mean_ref > thresh) & (wcf_now < (thresh*0.75))) | (wcf_now < (thresh*0.5)))

        # Define starting point for year from baseline
        trees_now = trees_ref.copy()
        # Remove trees where trees_gone criterion is met
        trees_now[trees_gone] = 0 

        
        # Step 5: Forest mapping
        forest_now = map_forest(trees_now, **forest_definitions[definition])
        
        # apply same kernel filter to the two tree height data sources
        forest_def_height = copy.deepcopy(forest_definitions)   # deepcopy required to copy all layers of keys
        forest_def_height['FAO']['min_crown_cover'] = 0
        forest_def_height['Australian']['min_crown_cover'] = 0

        # meta and zhao get updated in loop: create a copy to reserve the original version from input
        meta_theight_temp = meta_theight.copy()
        meta_theight_temp = meta_theight_temp.fillna(0).isel(time = 0).values
        zhao_theight_temp = zhao_theight.copy()
        zhao_theight_temp = zhao_theight_temp.isel(time = 0).values

        meta_forest = map_forest(meta_theight_temp>forest_def_height[definition]['min_height'], debug=True, **forest_def_height[definition])  # (1,256,256)
        zhao_forest = map_forest(zhao_theight_temp>forest_def_height[definition]['min_height'], **forest_def_height[definition])
        forest_h = (meta_forest + zhao_forest) > 0     #choose the larger of meta and zhao

        forest_now[forest_h == 0] = 0
        forests_year.append(forest_now)


    # Step 6: Combine reference and now trees and return as xarray DataArray
    return xr.DataArray(
        data=np.stack([trees_ref] + forests_year, axis=0).astype(np.float32),
        dims=['time', 'y', 'x'],
        coords={
            'time': [
                datetime.datetime(year=2020, month=1, day=1),
                datetime.datetime(year=2021, month=1, day=1),
                datetime.datetime(year=2022, month=1, day=1),
                datetime.datetime(year=2023, month=1, day=1)
            ],
            'y': da1.y,
            'x': da1.x
        }
    )