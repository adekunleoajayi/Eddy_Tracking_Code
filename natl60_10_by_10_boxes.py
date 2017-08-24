#   Author: Adekunle Opeoluwa Ajayi.
#   Institution: MEOM, Institut des Géosciences de l'Environnement (IGE),Université Grenoble Alpes,France.
#   Email: adekunle.ajayi@univ-grenoble-alpes.fr
#   Webpage:     https://adeajayi-kunle.github.io

# - Generate a 10 by 10 degree box to span north atlantic region.
import sys
import numpy as np
from netCDF4 import Dataset as ncopen

# - Define grid file
gridfile = '/Users/adekunle/Data/NATL60_Data/NA/NATL60_coordinates_v4.nc'

# - Define read data function
def read_datagrid(gridfile,latmin=None,latmax=None,lonmin=None,lonmax=None):
    """Return navlon,navlat."""
    ncfile = ncopen(gridfile,'r')
    # load navlon and navlat
    _navlon = ncfile.variables['nav_lon'][:,:]
    _navlat = ncfile.variables['nav_lat'][:,:]
    #-Define domain
    domain = (lonmin<_navlon) * (_navlon<lonmax) * (latmin<_navlat) * (_navlat<latmax)
    where = np.where(domain)
    vlats = _navlat[where]
    vlons = _navlon[where]
    #get indice
    jmin = where[0][vlats.argmin()]
    jmax = where[0][vlats.argmax()]
    imin = where[1][vlons.argmin()]
    imax = where[1][vlons.argmax()]
    #load arrays
    navlon = _navlon[jmin:jmax+1,imin:imax+1]
    navlat = _navlat[jmin:jmax+1,imin:imax+1]
    return navlon,navlat,jmin,jmax,imin,imax

# - Define box dimensions
Box_01 = ['40.0','30.0','-70.0','-60.0','Box_1']
Box_02 = ['40.0','30.0','-60.0','-50.0','Box_2']
Box_03 = ['40.0','30.0','-50.0','-40.0','Box_3']
Box_04 = ['40.0','30.0','-40.0','-30.0','Box_4']
Box_05 = ['40.0','30.0','-30.0','-20.0','Box_5']
Box_06 = ['40.0','30.0','-20.0','-10.0','Box_6']

Box_07 = ['50.0','40.0','-50.0','-40.0','Box_7']
Box_08 = ['50.0','40.0','-40.0','-30.0','Box_8']
Box_09 = ['50.0','40.0','-30.0','-20.0','Box_9']
Box_10 = ['50.0','40.0','-20.0','-10.0','Box_10']

Box_11 = ['60.0','50.0','-55.0','-45.0','Box_11']
Box_12 = ['60.0','50.0','-45.0','-35.0','Box_12']
Box_13 = ['60.0','50.0','-35.0','-25.0','Box_13']
Box_14 = ['60.0','50.0','-25.0','-15.0','Box_14']


# - Generate box array
box_arr = []
for ii in np.arange(1,15,1):
    name = eval('Box_'+str(ii).zfill(2))
    box_arr.append(name)

#- defining dictionaries for the boxes
class box: # empty container.
    def __init__(self,name=None):
        self.name = name
        return

dictboxes = {}

for ibox in box_arr:
    #print(ibox)
    y2 = eval(ibox[0]) ;y1 = eval(ibox[1]);
    x2 = eval(ibox[2]) ;x1 = eval(ibox[3]);
    box_name = ibox[4]
    
    # - Obtain navlon and Navlat
    navlon,navlat,jmin,jmax,imin,imax = read_datagrid(gridfile,latmin=y1,latmax=y2,lonmin=x2,lonmax=x1)
    
    # - save box parameter
    abox = box(box_name)
    abox.lonmin = navlon.min()
    abox.lonmax = navlon.max()
    abox.latmin = navlat.min()
    abox.latmax = navlat.max()
    abox.navlon = navlon
    abox.navlat = navlat
    abox.imin = imin
    abox.imax = imax
    abox.jmin = jmin
    abox.jmax = jmax
    dictboxes[box_name] = abox

boxes = dictboxes.values()
