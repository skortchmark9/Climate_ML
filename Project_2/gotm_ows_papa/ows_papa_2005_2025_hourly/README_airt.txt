
If you use these data in publications, please acknowledge the 
OCS Project Office of NOAA/PMEL.  Also, we would appreciate
receiving a preprint and/or reprint of publications utilizing 
the data for inclusion in the OCS bibliography.  Relevant 
publications should be sent to:

OCS Project Office
Dr. Meghan F. Cronin
NOAA/Pacific Marine Environmental Laboratory
7600 Sand Point Way NE
Seattle, WA 98115

Please send comments, questions, or problems to "Meghan.F.Cronin@noaa.gov"

=====================================================================
All data updates since January 2017 are tracked at the following URL:
https://www.pmel.noaa.gov/ocs/data/update_logs/Papa_Update_Log.htm
=====================================================================

The following topics will be covered in this readme file:
 
  1. Air Temperature
  2. Mooring Information for Papa
  3. Time Stamp
  4. Calculation of Averages
  5. Sampling and Sensors
  6. Quality Codes
  7. Source Codes

1. Air Temperature:

Air temperature is in units of degrees centigrade. Instrument 
height in the data files is shown as a negative depth.

2. Mooring Information for Papa: 

The Papa surface mooring site is nominally at 50.1°N, 144.9°W. 
As an active site from June 2007-present, the mooring is serviced 
each year, with new anchor positions that vary about the nominal 
location.  This taut line mooring has a watch circle radius of 
~3km. For users performing inter-comparisons, it may be important 
to use the actual position of the buoy from the GPS data. Depths 
shown in the delivered Papa files represent the location of 
the sensor on the mooring line.   

3. Time Stamp:

Time associated with data represent the sample time for single sample
values or the middle of the averaging interval for average values.  For
example, daily averages are computed starting at 0000 GMT and are
assigned an observation "time stamp" of 1200 GMT.

4. Calculation of Averages:

From the highest resolution data, Hourly (if applicable) and Daily 
Averages are calculated. 5-day, Monthly, and Quarterly Averages are 
computed if enough data are present at the previous resolution.  
Relevant definitions include:

Hourly: Average calculated if the highest resolution data are not 
already hourly.  High resolution data are passed through an 
X point Hanning filter, where X is:

5 if high resolution is 30min
7 if high resolution is 20min
13 if high resolution is 10min
60 if high resolution is 2min
120 if high resolution is 1min 

Over 50% (87.5% for SWR) of the points must be present to get an hourly 
value. Missing values are accounted for by dropping their weights from 
the calculation.

Daily: Average calculated using a boxcar filter on the high resolution 
data.  Over 50% (87.5% for SWR) of the points must be present to compute 
a daily value.  Missing values are accounted for using “nanmean,” which 
adds up the data, treating missing values as 0, and divides by the number 
of non-missing data points.

5-Day: Average of data collected during consecutive five day 
intervals. A minimum of 2 daily values are required to compute 
a 5-day average.

Monthly: Average of all the data collected during each month.
A minimum of 15 daily values are required to compute a monthly 
average.

Quarterly: Average of 3 monthly values. A minimum of 2 monthly 
values are required to compute a quarterly average. 12 quarterly 
averages are computed for each year, one for each center month, 
which includes the previous month, the center month, and the 
next month in the average.

5. Sampling and Sensors:

For detailed information about sampling and sensors, 
see these two web pages:

   http://www.pmel.noaa.gov/OCS/sampling-rates

   http://www.pmel.noaa.gov/OCS/sensors

6. Quality Codes:

In ascii format files organized by site, you will find data 
quality and source codes to the right of the data. In NetCDF 
format files organized by site, you will find quality and 
source variables with the same shape as the data.
These codes are defined below.

Using the quality codes you can tune your analysis to 
trade-off between quality and temporal/spatial coverage.
Quality code definitions are listed below

  0 = datum missing

  1 = highest quality; Pre/post-deployment calibrations agree to within
  sensor specifications.  In most cases only pre-deployment calibrations 
  have been applied

  2 = default quality; Pre-deployment calibrations applied.  Default
  value for sensors presently deployed and for sensors which were either 
  not recovered or not calibratable when recovered.

  3 = adjusted data; Pre/post calibrations differ, or original data do
  not agree with other data sources (e.g., other in situ data or 
  climatology), or original data are noisy. Data have been adjusted in 
  an attempt to reduce the error.

  4 = lower quality; Pre/post calibrations differ, or data do not agree
  with other data sources (e.g., other in situ data or climatology), or 
  data are noisy.  Data could not be confidently adjusted to correct 
  for error.

  5 = sensor or tube failed

7. Source Codes:

  0 - No Sensor, No Data 
  1 - Real Time (Telemetered Mode)
  2 - Derived from Real Time
  3 - Temporally Interpolated from Real Time
  4 - Source Code Inactive at Present
  5 - Recovered from Instrument RAM (Delayed Mode)
  6 - Derived from RAM
  7 - Temporally Interpolated from RAM
