# GOTM With Surface Forcings from OWS Papa

Source website: https://www.pmel.noaa.gov/ocs/data/disdel/
Install GOTM like so: https://gotm.net/software/linux/

Once installed, it can be run like so:

```
./gotm gotm.yaml
```

See `out.txt` for example output.

## Downloading files from the website, and then converting them to dat files.
```
# Air temperature
./papa_nc2dat.py -i $PAPA/airt50n145w_hr.cdf -o $PAPA/airt50n145w_hr.cdf.dat -ds 20080201 -de 20230101 -v AT_21
# Air pressure (note conversion to pascals * 100, also it doesnt go as far)
./papa_nc2dat.py -i $PAPA/bp50n145w_hr.cdf -o $PAPA/bp50n145w_hr.cdf.dat -ds 20080201 -de 20231201 -v BP_915
# Longwave Radiation
./papa_nc2dat.py -i $PAPA/lw50n145w_hr.cdf -o $PAPA/lw50n145w_hr.cdf.dat -ds 20080201 -de 20230101 -v Ql_136
# Shortwave Radiation
./papa_nc2dat.py -i $PAPA/rad50n145w_hr.cdf -o $PAPA/rad50n145w_hr.cdf.dat -ds 20080201 -de 20231201 -v RD_495
# Relative Humidity
./papa_nc2dat.py -i $PAPA/rh50n145w_hr.cdf -o $PAPA/rh50n145w_hr.cdf.dat -ds 20080201 -de 20230101 -v RH_910
# Sea Surface Salinity
./papa_nc2dat.py -i $PAPA/sss50n145w_hr.cdf -o $PAPA/sss50n145w_hr.cdf.dat -ds 20080201 -de 20230101 -v S_41
# Sea Surface Temperature
./papa_nc2dat.py -i $PAPA/sst50n145w_hr.cdf -o $PAPA/sst50n145w_hr.cdf.dat -ds 20080201 -de 20230101 -v T_25
# Sea Surface Density
./papa_nc2dat.py -i $PAPA/ssd50n145w_hr.cdf -o $PAPA/ssd50n145w_hr.cdf.dat -ds 20080201 -de 20230101 -v STH_71
# Wind U
./papa_nc2dat.py -i $PAPA/w50n145w_hr.cdf -o $PAPA/w50n145w_hr_u.cdf.dat -ds 20080201 -de 20230101 -v WU_422
# Wind V
./papa_nc2dat.py -i $PAPA/w50n145w_hr.cdf -o $PAPA/w50n145w_hr_v.cdf.dat -ds 20080201 -de 20230101 -v WV_423

### We decided not to use the profiles here to let the mixed layer evolve
### But they are recorded here for posterity.

# Temperature Profile
./papa_nc2dat.py -i $PAPA/t50n145w_hr.cdf -o $PAPA/t50n145w_hr.cdf.dat -ds 20080201 -de 20231201 -v T_20
# Density Profile
./papa_nc2dat.py -i $PAPA/d50n145w_hr.cdf -o $PAPA/d50n145w_hr.cdf.dat -ds 20080201 -de 20230101 -v STH_71
# Salinity Profile
./papa_nc2dat.py -i $PAPA/s50n145w_hr.cdf -o $PAPA/s50n145w_hr.cdf.dat -ds 20080201 -de 20230101 -v QS_5041
```
