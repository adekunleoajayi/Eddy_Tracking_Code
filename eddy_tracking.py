'''

  Software for the tracking of eddies in
  OFAM model output following Chelton et
  al., Progress in Oceanography, 2011.

'''

# Load required modules
import numpy as np
import eddy_functions as eddy

# Load parameters
from params import *
from natl60_10_by_10_boxes import boxes as rboxes

boxes = np.array([box.name for box in rboxes]);

for box in boxes:
    
    file_name = Summer_boxes+box+'.npz'
    print(file_name)
    data = np.load(file_name)
    det_eddies = data['eddies'] # len(eddies) = number of time steps

    # Initialize eddies discovered at first time step
    eddies = eddy.eddies_init(det_eddies)

    # Stitch eddy tracks together at future time steps
    rossrad = eddy.load_rossrad() # Atlas of Rossby radius of deformation and first baroclinic wave speed (Chelton et al. 1998)

    for tt in range(1, T):
        # Track eddies from time step tt-1 to tt and update corresponding tracks and/or create new eddies
        eddies = eddy.track_eddies(eddies, det_eddies, tt, dt, dt_aviso, dE_aviso, rossrad, eddy_scale_min, eddy_scale_max)
    # Add keys for eddy age and flag if eddy was still in existence at end of run
    for ed in range(len(eddies)):
        eddies[ed]['age'] = len(eddies[ed]['lon'])

    np.savez(Summer_boxes+'eddy_track_'+box, eddies=eddies)
