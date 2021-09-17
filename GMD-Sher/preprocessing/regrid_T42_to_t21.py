

import xarray as xr
from dask.diagnostics import ProgressBar
ProgressBar().register()


for model in ('puma', 'plasim'):

    path='/mnt/climstorage/sebastian/gcm_complexity_machinelearning/modelruns/preprocessed/'
    ifile = path+'/'+model+'t42reordered.merged.nc'
    ofile=path+'/'+model+'t42_regridt21reordered.merged.nc'

    data = xr.open_dataarray(ifile, chunks={'time':100})

    # bilinear regridding from T42 to T21 is the same as only retaining every second gridpoint
    regridded = data[:,::2,::2]

    regridded.to_netcdf(ofile)
