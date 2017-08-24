

#- Essential modules
import numpy as np
import os
import dask
import xarray as xr
import GriddedData
import time
from netCDF4 import Dataset
import matplotlib.cm as mplcm
import matplotlib.ticker as mticker

#- Other modules
import sys
import glob
import numpy.ma as ma
import scipy as sc
### quick plot
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

########################################
def mkdir(p):
    """make directory of path that is passed"""
    import os
    try:
        os.makedirs(p)
            print "output folder: "+p+ " does not exist, we will make one."
        except OSError as exc: # Python >2.5
            import errno
                if exc.errno == errno.EEXIST and os.path.isdir(p):
                    pass
                        else: raise


########################################
def Month_Index(Month):
    '''
    This function calls the numerical index of the months in the calender year
    '''
    Mon_Index = ['01','02','03','04','05','06','07','08','09','10','11','12']
    Mon_Name = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    for i in range(len(Mon_Index)):
        if Month == Mon_Name[i]:
            Index = Mon_Index[i]
    return Index

########################################
def Month_days(Month):
    '''
        This function calls the number of days of the months in the calender year
        '''
    Mon_days = [31,28,31,30,31,30,31,31,30,31,30,31]
    Mon_Name = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    for i in range(len(Mon_days)):
        if Month == Mon_Name[i]:
            days = Mon_days[i]
    return days


######### Define Month ########
YEAR = '2013'
MONTH = 'SEP'
MTH = Month_Index(MONTH)

#-Tracking
Tracked_boxes = '/Users/adekunle/Data/Eddy_Detection/NATL60_10_by_10/Tracked_boxes/'
Detected_boxes = '/Users/adekunle/Data/Eddy_Detection/NATL60_10_by_10/Detected_boxes/'
Winter_boxes = '/Users/adekunle/Data/Eddy_Detection/NATL60_10_by_10/Winter_boxes/'
Summer_boxes = '/Users/adekunle/Data/Eddy_Detection/NATL60_10_by_10/Summer_boxes/'

# - data pathway
database = "/Users/adekunle/Data/NATL60_Data/NA"
ssh_file = database + os.sep + 'NATL60-CJM165_y'+YEAR+'m'+MTH+'.1d.SSH.nc'
curl_file = database + os.sep + 'NATL60-CJM165_y'+YEAR+'m'+MTH+'.1d_CURLoverf.nc'

######### Open data ########
data_ssh = Dataset(ssh_file,mode='r')['sossheig']
data_curl = Dataset(curl_file,mode='r')['socurloverf']

########## working folder #########
data_dir = '/Users/adekunle/Data/Eddy_Detection/NATL60_10_by_10/Detected/'+YEAR+'_'+MTH+'/Analysis/'
plot_dir = data_dir + 'plots_'+YEAR+'_'+MTH+'/'
mkdir(data_dir)
mkdir(plot_dir)

######## Counter ###############
T = Month_days(MONTH) # Number of time steps to loop over
#T = 92 #457 # Number of time steps to loop over for eddy tracking

###########Fixed Parameter##################
res = 1.0/60 # horizontal resolution of SSH field [degrees]
dt = 1. # Sample rate of detected eddies [days]

ssh_crit_max = 1.0
dssh_crit = 0.01
ssh_crits = np.flipud(np.arange(-ssh_crit_max, ssh_crit_max+dssh_crit, dssh_crit))


######### Range of Eddy Pixel #########
Npix_min = 8 # min number of eddy pixels
Npix_max = 10000 # max number of eddy pixels

amp_thresh = 0.005 # minimum eddy amplitude [m]
d_thresh = 200. # max linear dimension of eddy [km] ;
dt_aviso = 1. # Sample rate used in Chelton et al. (2011) [days]
dE_aviso = 150. # Length of search ellipse to East of eddy used in Chelton et al. (2011) [km]
eddy_scale_min = 0.25 # min ratio of amplitude of new and old eddies
eddy_scale_max = 2.5 # max ratio of amplidude of new and old eddies
dt_save = 1 # Step increments at which to save data while tracking eddies
