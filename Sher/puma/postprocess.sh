pp_exec="/home/sebastian/simple_gcm_machinelearning/plasim/PlaSim_0318/postprocessor/burn7.x"

rundir="/climstorage/sebastian/plasim/testrun/"
#rundir="/climstorage/sebastian/plasim/"


for year in {001..999}; do
  ifile=${rundir}/testrun.${year}
  ofile=${ifile}_plevel.nc
  echo $ifile
  ${pp_exec} <pp_namelist_plevel.nl >pp_out -d $ifile $ofile
done
