'''

  Software for the tracking of eddies in
  OFAM model output following Chelton et
  al., Progress in Oceanography, 2011.

'''

# Load required modules
import numpy as np
import numpy.ma as ma
import matplotlib
# Turn the followin on if you are running on storm sometimes
#- Forces matplotlib to not use any Xwindows
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import eddy_functions as eddy

# Load parameters
from params import *
from natl60_10_by_10_boxes import boxes as rboxes

##########################################
def detect_eddies_for_boxes(rboxes,data_ssh,data_curl,days):
    for rbox in rboxes:
        print('processing box ' +  rbox.name)
        detect_eddies_for_box(rbox,data_ssh,data_curl,days)

##########################################
def detect_eddies_for_box(rbox,data_ssh,data_curl,days):
    valuesbox_ssh,valuesbox_curl = get_values_in_box(rbox,data_ssh,data_curl)
    find_eddies(valuesbox_ssh,valuesbox_curl,rbox.navlon,rbox.navlat,rbox.name,days=days)

##########################################
def get_values_in_box(rbox,data_ssh,data_curl):
    jmin,jmax = rbox.jmin,rbox.jmax
    imin,imax = rbox.imin,rbox.imax
    values_ssh = data_ssh[:,jmin:jmax+1,imin:imax+1]
    values_curl = data_curl[:,jmin:jmax+1,imin:imax+1]
    return values_ssh,values_curl

##########################################
def find_eddies(var_ssh,var_curl,navlon,navlat,boxname,days):
    lon_eddies_a = []
    lat_eddies_a = []
    amp_eddies_a = []
    area_eddies_a = []
    scale_eddies_a = []
    
    lon_eddies_c = []
    lat_eddies_c = []
    amp_eddies_c = []
    area_eddies_c = []
    scale_eddies_c = []
    
    for tt in range(days):
        ssh_data = np.array(var_ssh[tt])
        curl_data = np.array(var_curl[tt])
        # Load map of sea surface height (SSH) and Vorticity (CURL)
        eta,lon,lat = eddy.load_eta(ssh_data,navlon,navlat)
        curl,lon,lat = eddy.load_eta(curl_data,navlon,navlat)
        
        #print('Detecting anticyclonic eddies ........')
        ## Detect lon and lat coordinates of eddies using eta nad not eta_filt
        lon_eddies, lat_eddies, amp, area, scale = eddy.detect_eddies_scale_npix(eta, lon, lat, ssh_crits, res, Npix_min, Npix_max, amp_thresh, d_thresh, cyc='anticyclonic')
        lon_eddies_a.append(lon_eddies)
        lat_eddies_a.append(lat_eddies)
        amp_eddies_a.append(amp)
        area_eddies_a.append(area)
        scale_eddies_a.append(scale)
        
        #print('Detecting cyclonic eddies ........')
        lon_eddies, lat_eddies, amp, area, scale = eddy.detect_eddies_scale_npix(eta, lon, lat, ssh_crits, res, Npix_min, Npix_max, amp_thresh, d_thresh, cyc='cyclonic')
        lon_eddies_c.append(lon_eddies)
        lat_eddies_c.append(lat_eddies)
        amp_eddies_c.append(amp)
        area_eddies_c.append(area)
        scale_eddies_c.append(scale)
        
        # Plot map of filtered SSH field
        eddies_a=(lon_eddies_a[tt],lat_eddies_a[tt])
        eddies_c=(lon_eddies_c[tt],lat_eddies_c[tt])
        
        eddy.detection_plot(tt,YEAR,MTH,lon,lat,eta,eddies_a,eddies_c,'rawtoo',plot_dir,boxname,findrange=False)
        
        eddy.plot_curl(tt,YEAR,MTH,lon,lat,curl,eddies_a,eddies_c,'rawtoo',plot_dir,boxname,findrange=False)
    
    # Combine eddy information from all days into a list
    eddies = eddy.eddies_list(lon_eddies_a, lat_eddies_a, amp_eddies_a, area_eddies_a, scale_eddies_a, lon_eddies_c, lat_eddies_c, amp_eddies_c, area_eddies_c, scale_eddies_c)
    np.savez(data_dir+'eddy_detect_'+YEAR+'_'+MTH+'_'+boxname, eddies=eddies)
##########################################

#-Run program
detect_eddies_for_boxes(rboxes,data_ssh,data_curl,T)

