import os
import numpy as np
from io import StringIO, BytesIO
import xarray as xr
from google.cloud import storage
from onnxruntime import InferenceSession
import copy
import cv2 as cv
from skimage.morphology import remove_small_objects
import datetime
 

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
    x = cv.filter2D(x.astype(np.float32), -1, kernel, borderType=cv.BORDER_CONSTANT) >= (1 - min_crown_cover)
    x = remove_small_objects(x, min_size=int(100 * min_cover_area))  #trim forest
    x = remove_small_objects(~x, min_size=int(100 * min_cover_area))  # trim non-forest

    return ~x


    
def get_model():
    client = storage.Client()
    bucket = client.get_bucket('terrakio-models-eu')
    blob = bucket.blob('wcfrf_AU_v39.onnx')
    model = io.BytesIO()
    blob.download_to_file(model)
    model.seek(0)

    return InferenceSession(model.read(), providers=["CPUExecutionProvider"])



def forest20aus(da1, da2, da3, da4, da5, da6, da7, meta_theight, zhao_theight, model, definition='Australian', thresh = 0.04):
    # each da has shape (N_quarters = 4 * N_yrs, H, W)
    
    ### 1. Compute NDVI once at the beginning
    epsilon = 1e-8
    nir_band_idx = 3  
    red_band_idx = 0  
    
    # Calculate quarterly NDVI as a DataArray to maintain time coordinate
    ndvi = (da4 - da1) / (da4 + da1 + epsilon)
    ndvi = np.clip(ndvi, -1, 1)
    
    # Create annual median NDVI mask to apply on annual WCF predictions
    ndvi_annual_mask1 = (ndvi > 0.1).groupby('time.year').median(dim='time').rename({'year': 'time'})
    
    # Calculate min and max annual NDVI for final mask
    ndvimin = ndvi.min(dim='time')
    ndvimax = ndvi.max(dim='time')
    ndvi_mask2 = (ndvimin > 0.1) * ((ndvimax - ndvimin) < 0.5)
    

    ### 2. Process all bands including pre-computed NDVI
    s2_data = np.zeros((8, da1.time.size, da1.y.size, da1.x.size))
    
    # Insert all 8 bands da
    data_arrays = [da1, da2, da3, da4, da5, da6, da7]
    for i, da in enumerate(data_arrays):
        s2_data[i] = da.values

    s2_data[7] = ndvi.values
    

    ### 3. Predict WCF for each year
    quarters_per_year = 4
    nyrs = int(da1.time.size // quarters_per_year)
    print(f"Number of years in input: {nyrs}")
    
    predictions = np.zeros((nyrs, da1.y.size, da1.x.size))
    
    # Predict each year separately
    for year in range(nyrs):
        # Extract the data for this year (all 8 bands for 4 quarters)
        start_idx = year * quarters_per_year
        end_idx = start_idx + quarters_per_year
        year_data = s2_data[:, start_idx:end_idx, :, :]  # Shape: (8, 4, H, W)
        assert year_data.shape == (8, quarters_per_year, da1.y.size, da1.x.size), \
            f"Expected yearly input shape (8, {quarters_per_year}, {da1.y.size}, {da1.x.size}), got {year_data.shape}"
       
        # Reshape to (height, width, quarters, bands) then to (H*W, 4*8)
        year_data = np.transpose(year_data, (2, 3, 1, 0))  # Shape: (H, W, 4, 8)
        year_input = year_data.reshape(-1, quarters_per_year * 8)
        
        # prediction shape = (H*W,)
        year_output = model.run(None, {"X": year_input.astype(np.float32)})[0]
        
        # Reshape the output and store in predictions
        predictions[year] = year_output.reshape((da1.y.size, da1.x.size))
    
    # Now we have yearly predicted WCF
    wcf = xr.DataArray(
        data=predictions,  
        dims=['time', 'y', 'x'],
        coords={
            'time': da1.time[::4],  # assign the first quarter of each year as time
            'y': da1.y,
            'x': da1.x
        }
    )

    ### 4. Apply NDVI masks
    # Apply quarterly NDVI mask to yearly predictions
    wcf.values = np.where(ndvi_annual_mask1.values, wcf.values, 0.0)
    wcf_ref = wcf.sel(time=wcf.time.isin([2017, 2018, 2019, 2020]))

    ### 5. Crown cover mapping
    trees_ref = (wcf_ref > thresh).mean('time').values >= 0.5
    trees_ref = np.where(ndvi_mask2.values, trees_ref, 0.0)

    
    ### 6. Forest mapping & height filtering
    forest = map_forest(trees_ref, **forest_definitions[definition])

    forest_def_height = copy.deepcopy(forest_definitions)   # deepcopy required to copy all layers of keys
    forest_def_height['FAO']['min_crown_cover'] = 0
    forest_def_height['Australian']['min_crown_cover'] = 0

    meta_theight = meta_theight.fillna(0).isel(time = 0).values
    zhao_theight = zhao_theight.isel(time = 0).values

    meta_forest = map_forest(meta_theight>forest_def_height[definition]['min_height'], debug=True, **forest_def_height[definition])  
    zhao_forest = map_forest(zhao_theight>forest_def_height[definition]['min_height'], **forest_def_height[definition])
    forest_h = (meta_forest + zhao_forest) > 0     #choose the larger of meta and zhao

    forest[forest_h == 0] = 0  #(H,W)

    return xr.DataArray(
        data=forest[None, :, :],
        dims=['time', 'y', 'x'],
        coords={
            'time': [wcf.time.values[-1]],   #first quarter timestamp of the last input year
            'y': da1.y,
            'x': da1.x
        }
    )

    
