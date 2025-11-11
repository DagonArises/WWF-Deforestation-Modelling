import os
import numpy as np
import pandas as pd
import io
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
    x = cv.filter2D(x.astype(np.float32), -1, kernel, borderType=cv.BORDER_CONSTANT) > (1 - min_crown_cover)
    x = remove_small_objects(x, min_size=int(100 * min_cover_area))  #trim forest
    x = remove_small_objects(~x, min_size=int(100 * min_cover_area))  # trim non-forest

    return ~x

def get_model():
    client = storage.Client()
    bucket = client.get_bucket('terrakio-models-eu')
    blob = bucket.blob('fpcrf_AU_v4.onnx')
    model = BytesIO()
    blob.download_to_file(model)
    model.seek(0)
    
    return InferenceSession(model.read(), providers=["CPUExecutionProvider"])


def deforestation2024_aus(da1, da2, da3, da4, da5, da6, da7, meta_theight, zhao_theight, model, definition='Australian', thresh = 0.2):
    # each da has shape (N_quarters = 4 * N_yrs, H, W)
    
    ### 1. Compute NDVI once at the beginning
    epsilon = 1e-8
    ndvi = (da4 - da1) / (da4 + da1 + epsilon)
    ndvi = np.clip(ndvi, -1, 1)
    
    # Create annual median NDVI mask to apply on annual FPC predictions
    ndvi_annual_mask1 = (ndvi > 0.1).groupby('time.year').median(dim='time').rename({'year': 'time'})
    
    # Calculate min and max annual NDVI for final mask
    ndvimin = ndvi.min(dim='time')
    ndvimax = ndvi.max(dim='time')
    ndvi_mask2 = (ndvimin > 0.1) * ((ndvimax - ndvimin) < 0.5)
    

    ### 2. Process all bands including pre-computed NDVI
    s2_data = np.zeros((7, da1.time.size, da1.y.size, da1.x.size))
    
    # Insert all 7 bands da
    data_arrays = [da1, da2, da3, da4, da5, da6, da7]
    for i, da in enumerate(data_arrays):
        s2_data[i] = da.values
    

    ### 3. Predict FPC for each year
    quarters_per_year = 4
    nyrs = int(da1.time.size // quarters_per_year)
    print(f"Number of years in input: {nyrs}")
    
    predictions = np.zeros((nyrs, da1.y.size, da1.x.size))
    
    # Predict each year separately
    for year in range(nyrs):
        # Extract the data for this year (all 7 bands for 4 quarters)
        start_idx = year * quarters_per_year
        end_idx = start_idx + quarters_per_year
        year_data = s2_data[:, start_idx:end_idx, :, :]  # Shape: (7, 4, H, W)
        assert year_data.shape == (7, quarters_per_year, da1.y.size, da1.x.size), \
            f"Expected yearly input shape (7, {quarters_per_year}, {da1.y.size}, {da1.x.size}), got {year_data.shape}"
       
        # Reshape to (height, width, quarters, bands) then to (H*W, 4*7)
        year_data = np.transpose(year_data, (2, 3, 1, 0))  # Shape: (H, W, 4, 7)
        year_input = year_data.reshape(-1, quarters_per_year * 7)
        
        # prediction shape = (H*W,)
        year_output = model.run(None, {"X": year_input.astype(np.float32)})[0]
        
        # Reshape the output and store in predictions
        predictions[year] = year_output.reshape((da1.y.size, da1.x.size))
    
    # Now we have yearly predicted FPC
    fpc = xr.DataArray(
        data=predictions,  
        dims=['time', 'y', 'x'],
        coords={
            'time': da1.time[::4],  # assign the first quarter of each year as time
            'y': da1.y,
            'x': da1.x
        }
    )


    ### 4. Apply NDVI masks
    # Apply yearly NDVI mask to yearly predictions
    fpc.values = np.where(ndvi_annual_mask1.values, fpc.values, 0.0)
    fpc_ref = fpc.sel(time=fpc.time.dt.year.isin([2017, 2018, 2019,2020]))

    ### 5. Crown cover mapping
    trees_ref = (fpc_ref > thresh).mean('time').values >= 0.5
    trees_ref = np.where(ndvi_mask2.values, trees_ref, 0.0)

    mean_ref = fpc_ref.mean('time').values        
    std_ref = fpc_ref.std('time').values  
    

    ### 5. Create relative deforestation maps for each year
    input_years = [2024, 2023, 2022, 2021]
    year_maps = []
    
    for yr in input_years:
        fpc_now = fpc.sel(time=fpc.time.dt.year==yr).values
        z = (fpc_now - mean_ref) / std_ref          

        # Crown cover mapping
        trees_gone = (((z < -3) & (mean_ref > thresh) & (fpc_now < (thresh*0.75))) | (fpc_now < (thresh*0.5)))
        trees_gone = trees_gone.squeeze()
        trees_now = trees_ref.copy()
        trees_now[trees_gone] = 0                     # Remove trees where TreeGone criterion is met.

        # Forest mapping
        forest_ref = map_forest(trees_ref, **forest_definitions[definition])
        forest_now = map_forest(trees_now, **forest_definitions[definition])
        
        # Apply height filtering
        forest_def_height = copy.deepcopy(forest_definitions)
        forest_def_height['FAO']['min_crown_cover'] = 0
        forest_def_height['Australian']['min_crown_cover'] = 0

        meta_theight_temp = meta_theight.copy().fillna(0).isel(time=0).values
        zhao_theight_temp = zhao_theight.copy().isel(time=0).values

        meta_forest = map_forest(meta_theight_temp > forest_def_height[definition]['min_height'], 
                                debug=True, **forest_def_height[definition])
        zhao_forest = map_forest(zhao_theight_temp > forest_def_height[definition]['min_height'], 
                                **forest_def_height[definition])
        forest_h = (meta_forest + zhao_forest) > 0
        
        forest_ref[forest_h == 0] = 0 
        forest_now[forest_h == 0] = 0
        
        # Initial deforestation map (0: no forest, 1: forest, 2: deforested in this year)
        deforest = forest_ref.astype(np.float32)
        deforest[(forest_ref == 1) & (forest_now == 0)] = 2
        
        year_maps.append(deforest)
    
    
    
    # Start with the most recent year's map
    final_map = year_maps[0].copy()
    num_years = len(input_years)
    
    # First, convert encoding to match desired output format:
    # 0 -> 255 (no forest)
    # 1 -> 0 (forest)
    # 2 -> N (deforested in most recent year, where N = number of years)
    final_map[final_map == 0] = 255  # No forest
    final_map[final_map == 1] = 0    # Forest
    final_map[final_map == 2] = num_years  # Deforested in most recent year
    
    # The value for deforestation in the most recent year is num_years 
    # checking for continuous deforestation in earlier years
    for i in range(1, num_years):
        # Get the current year's deforestation map
        current_defor_map = year_maps[i]
        
        # The current deforestation value we're looking for in the final map
        current_value_in_final = num_years - (i - 1)
        
        # The value we'll assign if this pixel is also deforested in the current year
        new_value_if_continuous = num_years - i
        
        # Update the final map for continuously deforested pixels
        mask = (final_map == current_value_in_final) & (current_defor_map == 2)
        final_map[mask] = new_value_if_continuous
    
    # Return as a DataArray with the latest timestamp
    return xr.DataArray(
        data=final_map.reshape((1, *da1.shape[1:])),
        dims=['time', 'y', 'x'],
        coords={
            'time': [fpc.time.values[-1]],
            'y': da1.y,
            'x': da1.x
        }
    )
