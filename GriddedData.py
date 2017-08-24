#!/usr/bin/env  python
#=======================================================================
"""GriddedData.py
Define grid object and interpolation tools for working with gridded data.
Some basic plotting facility is also provided.
"""
#=======================================================================


import numpy
import numpy as npy
N = npy
np = npy
import numpy.ma as ma
from mpl_toolkits import basemap

dint = npy.int8
dfloat = npy.float32

grav = 9.81                  # acceleration due to gravity (m.s-2)
omega = 7.292115083046061e-5 # earth rotation rate (s-1)
earthrad = 6371229            # mean earth radius (m)
deg2rad = npy.pi / 180.

mod360 = lambda x: npy.mod(x+180,360)-180
#========================= Grid Class ===================================
class grid2D:
   """Two-dimensional grid object with the associated operators.
   This class is based on NEMO ocean model notations and operators. 
   """
   def __init__(self,navlat=None,navlon=None,mask=None):
       """Initialize grid object from navlon,navlat arrays
       """
       if (len(navlat.shape)==1 and len(navlon.shape)==1) or (navlon.shape!=navlat.shape): 
          navlon, navlat = npy.meshgrid(navlon,navlat)
       #
       self.navlon = navlon
       self.navlat = navlat
       self.tmask = mask
       self.jpj,self.jpi = navlon.shape
       self.jpk = 1
       self.depthT = [0]
       self.shape = (self.jpj,self.jpi)
       self._get_gphiglam()
       self._get_scalefactors()
       self._get_masks()
       self._get_surf()


   def _get_gphiglam(self):
       """Get glam*,gphi* for * in t,u,v,f  
       """
       self.glamt = self.navlon
       self.gphit = self.navlat
       self.glamu = (self.glamt + npy.roll(self.glamt,-1,axis=-1))/2.
       self.gphiu = (self.gphit + npy.roll(self.gphit,-1,axis=-1))/2.
       self.glamv = (self.glamt + npy.roll(self.glamt,-1,axis=-2))/2.
       self.gphiv = (self.gphit + npy.roll(self.gphit,-1,axis=-2))/2.
       self.glamf = (self.glamu + npy.roll(self.glamu,-1,axis=-2))/2.
       self.gphif = (self.gphiu + npy.roll(self.gphiu,-1,axis=-2))/2.


   def _get_scalefactors(self,method='1'):
       """Get the scale factors (m) : e1*,e2* for * in t,u,v,f  
       """
       
       for gtype in ['t','u','v','f']:
          lam = eval('self.glam' + gtype)
	  phi = eval('self.gphi' + gtype)
          djlam,dilam = npy.gradient(lam)
          djphi,diphi = npy.gradient(phi)
          e1 = earthrad * deg2rad * npy.sqrt( (dilam * npy.cos(deg2rad * phi))**2. + diphi**2.)
	  e2 = earthrad * deg2rad * npy.sqrt( (djlam * npy.cos(deg2rad*phi))**2. + djphi**2.)
	  exec('self.e1' + gtype + ' = e1')
	  exec('self.e2' + gtype + ' = e2')

   def _get_masks(self):
       """Get t,u,v,f-masks 
       """
       if (self.tmask is None):
          self.tmask = npy.ones(self.shape,dtype=dint)
          self.umask = self.tmask
          self.vmask = self.tmask
          self.fmask = self.tmask
       else:
          jpj,jpi = self.shape
          self.tmask = npy.array(self.tmask,dtype=dint)
          bigtmask = npy.ones((jpj+1,jpi+1),dtype=dint)
          bigtmask[0:jpj,0:jpi] = self.tmask
          self.umask= bigtmask[0:jpj,0:jpi] * bigtmask[0:jpj,1:jpi+1]
          self.vmask= bigtmask[0:jpj,0:jpi] * bigtmask[1:jpj+1,0:jpi]
          self.fmask= bigtmask[0:jpj,0:jpi] * bigtmask[0:jpj,1:jpi+1]\
                    * bigtmask[1:jpj+1,0:jpi] * bigtmask[1:jpj+1,1:jpi+1]

   def _get_surf(self):
       """Compute array surfaces.
       """
       self.u_surf = self.e2u * self.e1u
       self.v_surf = self.e2v * self.e1v
       self.f_surf = self.e2f * self.e1f
       self.t_surf = self.e2t * self.e1t
       self.surface = npy.sum(npy.sum(self.tmask * self.t_surf,axis=-1),axis=-1)


   def _get_corio(self):
       """Compute coriolis parameter on the grid
       """
       #coriogrid = 'corio_' + grid
       for grid in ['u','v','t']:
          coriogrid = 'corio_' + grid
          if not(hasattr(self,coriogrid)):
             exec('self.' + coriogrid + '= corio(self,grid=grid)')

          
#---------------------------- Masking ---------------------------------------
   def set_mask(self,nq,mask,msk_value=1.E20):
       """Set a mask on an array.
       """
       if ma.isMaskedArray(nq):
          nq=npy.array(nq,subok=True)
       q = nan_to_zero(nq)
       a_msk = abs(mask-1)
       a_msk = a_msk * msk_value
       mq = q * mask
       mq+= a_msk
       #
       return mq

#---------------------------- Grid Swapping ---------------------------------
#- Core swapping utilities
   def _gridi2iplus(self,var,mvol):
       jpi = self.jpi
       mvar =  mvol
       tabvar = 0.*var#npy.core.ma.masked_equal(0.*var,1)
       # newval(i) is (val(i) + val(i+1)) / 2
       tabvar[...,0:jpi-1] = mvar[...,0:jpi-1] * var[...,0:jpi-1]\
                            + mvar[...,1:jpi] * var[...,1:jpi]
       tabvar[...,0:jpi-1]/= mvar[...,0:jpi-1] + mvar[...,1:jpi]
       tabvar[...,jpi-1] = var[...,jpi-1]
       #
       return nan_to_mskval(npy.array(tabvar,subok=True))

   def _gridi2iminus(self,var,mvol):
       jpi = self.jpi
       mvar = mvol
       tabvar = 0.*var#npy.core.ma.masked_equal(0.*var,1)
       # newval(i) is (val(i-1) + val(i)) / 2
       tabvar[...,:,1:jpi] = mvar[...,:,0:jpi-1] * var[...,:,0:jpi-1]\
                           + mvar[...,:,1:jpi]   * var[...,:,1:jpi]
       tabvar[...,1:jpi]/= mvar[...,0:jpi-1] + mvar[...,1:jpi]
       tabvar[...,0] = var[...,0]
       #
       return nan_to_mskval(npy.array(tabvar,subok=True))

   def _gridj2jplus(self,var,mvol):
       jpj = self.jpj
       mvar = mvol
       tabvar = 0.*var#npy.core.ma.masked_equal(0.*var,1)
       # newval(j) is (val(j) + val(j+1)) / 2
       tabvar[...,0:jpj-1,:] = mvar[...,0:jpj-1,:] * var[...,0:jpj-1,:]\
                             + mvar[...,1:jpj,:]   * var[...,1:jpj,:]
       tabvar[...,0:jpj-1,:]/= mvar[...,0:jpj-1,:] + mvar[...,1:jpj,:]
       tabvar[...,jpj-1,:] = var[...,jpj-1,:]
       return nan_to_mskval(npy.array(tabvar,subok=True))


   def _gridj2jminus(self,var,mvol):
       jpj = self.jpj
       mvar = mvol
       tabvar = 0.*var#npy.core.ma.masked_equal(0*var,1)
       # newval(j) is (val(j-1) + val(j)) / 2
       tabvar[...,1:jpj,:] = mvar[...,0:jpj-1,:] * var[...,0:jpj-1,:]\
                           + mvar[...,1:jpj,:]   * var[...,1:jpj,:]
       tabvar[...,1:jpj,:]/= mvar[...,0:jpj-1,:] + mvar[...,1:jpj,:]
       tabvar[...,0,:] = var[...,0,:]
       return nan_to_mskval(npy.array(tabvar,subok=True))

   def _grid_2_grid_iright_jleft(self,var,mvol,mask):
       var1 = self._gridi2iplus(var,mvol)
       mvol1 = self._gridi2iplus(mvol,mask)
       var2 = self._gridj2jminus(var1,mvol1)
       return var2

   def _grid_2_grid_ileft_jright(self,var,mvol,mask):
       var1 = self._gridi2iminus(var,mvol)
       mvol1 = self._gridi2iminus(mvol,mask)
       var2 = self._gridj2jplus(var1,mvol1)
       return var2

#- User swapping utilities
   def gridf_2_gridT(self,w):
       """Return w (gridf) on gridT
       """
       mvol = self.f_surf
       msk = self.fmask
       w1 = self._gridj2jminus(w,mvol)
       mvol1 = self._gridj2jminus(mvol,msk)
       w2 = self._gridi2iminus(w1,mvol1)
       return w2

   def gridV_2_gridU(self,v):
       return self._grid_2_grid_iright_jleft(v,self.v_surf,self.vmask)

   def gridU_2_gridV(self,u):
       return self._grid_2_grid_ileft_jright(u,self.u_surf,self.umask)

   def gridT_2_gridV(self,v):
       """Return v (gridT) on gridV."""
       return self._gridj2jplus(v,self.t_surf)
    
   def gridT_2_gridU(self,u):
       """Return u (gridT) on gridU."""
       return self._gridi2iplus(u,self.t_surf)

   def gridU_2_gridT(self,u):
        """Return u (gridU) on gridT
        """
        return self._gridi2iminus(u,self.u_surf)

   def gridV_2_gridT(self,v):
       """Return v (gridV) on gridT
       """
       return self._gridj2jminus(v,self.v_surf)

#---------------------------- Vector Operators -----------------------------------
   def lamV(self,lam,V):
       """
       Return lambda * V.
        input 
        -lam : T-grid
        -V : U,V,W grid
       
       output 
        -lamV : U,V,W grid 
       """
       lamx = self.gridT_2_gridU(lam)
       lamVx = lamx * V[0]
       lamy = self.gridT_2_gridV(lam)
       lamVy = lamy * V[1]
       return lamVx,lamVy

   def dot(self,a,b,stag_grd=False):
       """
       Return the dot product a.b.
       -----------------------------
       input :
          ax,bx : grid U
          ax,by : grid V
       output :
          p : grid T
       """
       #
       ma1 = self.gridU_2_gridT(a[0])
       mb1 = self.gridU_2_gridT(b[0])
       ma2 = self.gridV_2_gridT(a[1])
       mb2 = self.gridV_2_gridT(b[1])
       p = ma1 * mb1 + ma2 * mb2
       return p

#---------------------------- Finite Differences ---------------------------------
   def d_i(self,q,partial_steps=None):
       """Return difference q(i+1)-q(i)
       """
       jpi=self.jpi
       di= q[...,1:jpi]-q[...,0:jpi-1]
       return di

   def d_j(self,q,partial_steps=None):
       """Return difference q(j+1)-q(j)
       """
       jpj=self.jpj
       dj=q[...,1:jpj,:]-q[...,0:jpj-1,:]
       return dj

   def m_i(self,q):
       """Return the average of q(i+1) and q(i)
       """
       #
       jpi=self.jpi
       mi= q[...,1:jpi]+q[...,0:jpi-1]
       mi/=2.
       return mi

   def m_j(self,q):
       """Return the average of q(j+1) and q(j)
       """
       #
       jpj=self.jpj
       mj=q[...,1:jpj,:]+q[...,0:jpj-1,:]
       mj/=2.
       return mj

   def setBC(self,q,axis,lim,msk_value=1E20):
       """Extends an array to fit the initial grid."""
       BC_shape=npy.array(q.shape,dtype=dfloat,subok=True) # subok is probably useles...
       if axis=='i':
           nax=-1
       elif axis=='j':
           nax=-2
       elif axis=='k':
           nax=-3
       #
       BC_shape[nax] = 1
       BC = N.ones(BC_shape) * msk_value
       #
       if lim==-1:
           new_q=npy.concatenate((q,BC),axis=nax)
       elif lim==1:
           new_q=npy.concatenate((BC,q),axis=nax)
       #
       return new_q


   def grad(self,q,masked=False):
       """
       Return the 2D gradient of a scalar field.
       input :  on T grid
       output : on U,V  grid
        """
       #
       jpj,jpi = self.shape
       #
       gx=self.d_i(q)
       gx/=self.e1u[:,0:jpi-1]
       #
       gy=self.d_j(q)
       gy/=self.e2v[0:jpj-1,:]
       #
       Bgx=self.setBC(gx,'i',-1)
       Bgy=self.setBC(gy,'j',-1)
       #
       if masked:
          Bgx=self.set_mask(Bgx,self.umask)
          Bgy=self.set_mask(Bgy,self.vmask)
       return Bgx,Bgy
       

   def matrixgradient(self,u,v,masked=False):
      """Return the 2d tensor of the gradient of a vector field.
         ux,vy : at t-points
         uy,vx : at f-points
      """
      jpj,jpi = self.shape
      ux = self.d_i(self.e2u * u)[...,:,:] / (self.e1t*self.e2t)[...,:,1:jpi]
      ux = self.setBC(ux,'i',-1) # t-point
      vy = self.d_j(self.e1v * v)[...,:,:] / (self.e1t*self.e2t)[...,1:jpj,:]
      vy = self.setBC(vy,'j',-1) # t-point
      uy = self.d_j(self.e1u * u)[...,:,:] / (self.e1f*self.e2f)[...,0:jpj-1,:] 
      uy = self.setBC(uy,'j',-1) # f-point 
      vx = self.d_i(self.e2v * v)[...,:,:] / (self.e1f*self.e2f)[...,:,0:jpi-1]
      vx = self.setBC(vx,'i',-1) # f-point
      if masked:
         ux = self.set_mask(ux,self.tmask)
         vy = self.set_mask(vy,self.tmask)
         vx = self.set_mask(vx,self.fmask)
         uy = self.set_mask(uy,self.fmask)
      return {'ux':ux,'vy':vy,'uy':uy,'vx':vx}


   def curl(self,a,masked=False):
       """Return the vertical component of the curl of a vector field.
       """
       #
       a1 = a[0]
       a2 = a[1]
       #
       jpi = self.jpi
       jpj = self.jpj
       #
       cz = ( self.d_i(self.e2v*a2)[...,0:jpj-1,:]\
            - self.d_j(self.e1u*a1)[...,:,0:jpi-1] )
       cz/= (self.e1f*self.e2f)[...,0:jpj-1,0:jpi-1]
       #
       Bcz = self.setBC(self.setBC(cz,'i',-1),'j',-1)
       #
       if masked:
          Bcz = self.set_mask(Bcz,self.fmask)
       return Bcz

   def div(self,a,masked=False):
       """
       Return the 2D divergence of a vector field.
       input :  grid U,V
       output : grid T
       """
       # 
       a1=a[0]
       a2=a[1]
       #
       jpi=self.jpi
       jpj=self.jpj
       #
       d=self.d_i(self.e2u*a1)[...,1:jpj,:]+self.d_j(self.e1v*a2)[...,:,1:jpi]
       d/=(self.e1t*self.e2t)[...,1:jpj,1:jpi]
       #
       Bd=self.setBC(self.setBC(d,'i',1),'j',1)
       #
       if masked:
          Bd = self.set_mask(Bd,self.tmask)
       #
       return Bd

   def shear_strain(self,a,masked=False):
       """Return the rate of shear strain r = vx + uy on the f-grid.
       """
       a1=a[0]
       a2=a[1]
       #
       jpi=self.jpi
       jpj=self.jpj
       # 
       r = ( self.d_i(self.e2v*a2)[...,0:jpj-1,:]\
            + self.d_j(self.e1u*a1)[...,:,0:jpi-1] )
       r/= (self.e1f*self.e2f)[...,0:jpj-1,0:jpi-1]
       #
       Br = self.setBC(self.setBC(r,'i',-1),'j',-1)
       if masked:
          Br = self.set_mask(Br,self.fmask)
       return Br

   def normal_strain(self,uv,masked=False):
       """Return the normal rate of  strain a = ux - vy on the T-grid.
       """
       u=uv[0]
       v=uv[1]
       #
       jpi=self.jpi
       jpj=self.jpj
       #
       a = self.d_i(self.e2u*u)[...,1:jpj,:] - self.d_j(self.e1v*v)[...,:,1:jpi]
       a/=(self.e1t*self.e2t)[...,1:jpj,1:jpi]
       Ba = self.setBC(self.setBC(a,'i',1),'j',1)
       if masked:
          Ba = self.set_mask(Ba,self.tmask)
       return Ba

   def ssh2uv(self,ssh):
       """Return u,v from sea surfac height on the grid
       """
       self._get_corio()
       hx,hy = self.grad(ssh)
       gf_u = grav / self.corio_u
       gf_u[npy.where(npy.abs(self.gphiu)<5.)] = 0
       gf_v = grav / self.corio_v
       gf_v[npy.where(npy.abs(self.gphiv)<5.)] = 0
       u = - gf_u * self.gridV_2_gridU(hy)
       v =   gf_v * self.gridU_2_gridV(hx)
       return u,v
#------------------------ Specific Grids------------------------------------

def gridAVISO_onethird():
    """Return a grid object corresponding to AVISO 1/3 global MERCATOR grid.
    """
    import IoData
    lat,lon = IoData.getAVISOlatlon()
    grd = grid2D(navlon=lon,navlat=lat)
    return grd

def gridAVISO_qd():
    """Return a grid object corresponding to AVISO 1/3 global qd grid.
    """
    import IoData
    lat,lon = IoData.getAVISOlatlon_qd()
    grd = grid2D(navlon=lon,navlat=lat)
    return grd

def gridNOAA_onequarter():
    """Return a grid object corresponding to NCDC/NOAA 1/4 global grid.
    """
    import IoData
    lat,lon,mask = IoData.getNOAAlatlonmask()
    grd = grid2D(navlon=lon,navlat=lat,mask=mask)
    return grd

#====================== Interpolation ======================================
class stdRegridder:
   """bilinear interpolation with basemap.interp. assumes the grid is rectangular.
   """
   def __init__(self,xin=None,yin=None,xout=None,yout=None,method='basemap'):
       self.xin = xin[0,:]
       self.yin =  yin[:,0]
       self.xout = xout
       self.yout = yout
       self.method = method

   def __call__(self,array):
       masked = ma.is_masked(array)
       if self.method is 'basemap':
          return basemap.interp(array, self.xin, self.yin, self.xout, self.yout, checkbounds=False, masked=masked, order=1)
       elif self.method is 'scipy':
          import scipy.interpolate
          interp = scipy.interpolate.interp2d(self.xin, self.yin, array, kind='linear')
          a1d = interp(self.xout[0,:],self.yout[:,0])
          return npy.reshape(a1d,self.yout.shape)

def grdRegridder(grdin=None,grdout=None,grdintype='t',grdouttype='t'):
    """Return a regridder based on grd instances.
    """
    xin = eval('grdin.glam' + grdintype)
    yin = eval('grdin.gphi' + grdintype)
    xout = eval('grdout.glam' + grdouttype)
    yout = eval('grdout.gphi' + grdouttype)
    return stdRegridder(xin=xin,yin=yin,xout=xout,yout=yout)

#====================== Carsening ======================================

def boxcar_factor_test(array2D,icrs=3,jcrs=3):
    """Test whether the shape of array2D is suited to coarsening with icrs,jcrs
    """
    jpj, jpi = array2D.shape
    if jpj%jcrs==0 and jpi%icrs==0:
       return True
    else:
       return False

def boxcar_reshape(array2D,icrs=3,jcrs=3):
    """Return a 3D array where values in boxes added in extra dimensions 
    """
    if not(boxcar_factor_test(array2D,icrs=icrs,jcrs=jcrs)):
       print "shape and coarsening factors are not compatible"
       return
    jpj, jpi = array2D.shape
    # target shape is shape = (jcrs, icrs, jpj/jcrs, jpi/icrs)
    t = np.reshape(array2D,(jpj,-1,icrs)) 		# (jpj, jpi/icrs, icrs)
    tt = t.swapaxes(0,2)				# (icrs,jpi/icrs, jpi)
    ttt = np.reshape(tt,(icrs,jpi/icrs,-1,jcrs)) 	# (icrs,jpi/icrs,jpj/jcrs, jcrs)
    tttt = ttt.swapaxes(1,3)  				# (icrs,jcrs,jpj/jcrs, jpi/icrs)
    ttttt = tttt.swapaxes(0,1) 				# (jcrs,icrs,jpj/jcrs, jpi/icrs)
    return ttttt

def boxcar_ravel(array2D,icrs=3,jcrs=3):
    """Return a 3D array where values in boxes are broadcasted along the third axis.
        output shape is (icrs*jcrs,jpj_crs,jpi_csr)
    """
    if not(boxcar_factor_test(array2D,icrs=icrs,jcrs=jcrs)):
       print "shape and coarsening factors are not compatible"
       return
    reshaped = boxcar_reshape(array2D,icrs=icrs,jcrs=jcrs)
    dum,dum,jpj,jpi = reshaped.shape
    raveled = reshaped.reshape((icrs*jcrs,jpj,jpi))
    return raveled

def boxcar_deep_ravel(array2D,icrs=3,jcrs=3):
    """Return a 3D array are
        output shape is (jpj_crs*jpi_csr,jcrs,icrs) 
    """
    if not(boxcar_factor_test(array2D,icrs=icrs,jcrs=jcrs)):
       print "shape and coarsening factors are not compatible"
       return
    reshaped = boxcar_reshape(array2D,icrs=icrs,jcrs=jcrs)
    dum,dum,jpj,jpi = reshaped.shape
    deep_raveled = reshaped.reshape((jcrs,icrs,jpj*jpi))
    deep_raveled = np.rollaxis(deep_raveled,2)
    return deep_raveled

def boxcar_sum(array2D,icrs=3,jcrs=3):
    """Return an array with values corresponding to sums of array2D within boxes. 
    """
    if not(boxcar_factor_test(array2D,icrs=icrs,jcrs=jcrs)):
       print "shape and coarsening factors are not compatible"
       return
    jpj, jpi = array2D.shape
    shape = (jpj/jcrs, jpi/icrs)
    sum_array = boxcar_ravel(array2D,icrs=icrs,jcrs=jcrs).sum(axis=0)
    return sum_array

class grdCoarsener:
    """Return a method that implements coarsening for a given input grid. 
    """
    def __init__(self,grdin,x_offset=0,y_offset=0,crs_factor=3):
        # loading
        self.fine_grid = grdin
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.crs_factor = crs_factor
        self.fine_shape = grdin.shape
        self.crs_factor = crs_factor
        # indices
        jpj,jpi = self.fine_shape
        jcrs, icrs = crs_factor, crs_factor
        jsize = jpj - (jpj - y_offset) % jcrs #- y_offset
        isize = jpi - (jpi - x_offset) % icrs #- x_offset
        self.crs_shape = ( isize / jcrs , isize / icrs )
        self.cut_array = lambda array2D:array2D[...,y_offset:jsize,x_offset:isize]
        self.weights = self.cut_array(self.fine_grid.t_surf)
        self.crs_area = boxcar_sum(self.weights,icrs=self.crs_factor,jcrs=self.crs_factor)
        self.crs_shape = self.crs_area.shape

    def __call__(self,array2D):
        cut_array2D = self.cut_array(array2D)
        bxc = lambda a:boxcar_sum(a,icrs=self.crs_factor,jcrs=self.crs_factor) 
        return bxc(cut_array2D * self.weights) / self.crs_area
        
    def return_ravel(self,array2D):
        cut_array2D = self.cut_array(array2D)
        rvl = lambda a:boxcar_ravel(a,icrs=self.crs_factor,jcrs=self.crs_factor)
        return rvl(cut_array2D)

    def return_deep_ravel(self,array2D):
        # invers with reshape(array2D.shape)
        cut_array2D = self.cut_array(array2D)
        rvl = lambda a:boxcar_deep_ravel(a,icrs=self.crs_factor,jcrs=self.crs_factor)
        return rvl(cut_array2D)


#====================== Coriolis ===========================================
def corio(dom,grid='t'):
    """Return Coriolis parameter.
    """
    exec('lat = dom.gphi' + grid)
    f = 2.*omega*npy.sin(lat*deg2rad)
    return f

def beta(dom,grid='t'):
    """Return planetary beta.
    """
    exec('lat = dom.gphi' + grid)
    beta = 2.*omega*npy.cos(lat*deg2rad) / earthrad
    return beta


#====================== Miscellaneous ======================================
def nan_to_zero(gz,max_val=1E20):
    """."""
    cgz = numpy.nan_to_num(gz)
    cgz[numpy.where(numpy.abs(cgz)>=max_val)]=0
    return cgz

def nan_to_mskval(gz,mskval=1.E20):
    tmp = -9E9*npy.pi
    lgz = gz.copy()
    lgz[npy.where(lgz==0.)]=tmp
    lgz = npy.nan_to_num(lgz)
    lgz[npy.where(lgz==0.)]=mskval
    lgz[npy.where(lgz==tmp)]=0.
    return lgz



#==================
# from http://wiki.scipy.org/Cookbook/SignalSmooth
def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    from pylab import mgrid
    from numpy import exp
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = mgrid[-size:size+1, -sizey:sizey+1]
    g = exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    from scipy import signal
    g = gauss_kern(n, sizey=ny)
    improc = signal.convolve(im,g, mode='same')
    return(improc)
