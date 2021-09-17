import xarray as xr

file = "C:/Users/Aditya/Desktop/Sher Code/puma_sample_30year_normalized.nc"
data = xr.open_dataarray(file, chunks={'time':1})

print(data)