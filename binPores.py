#programme to bin pore stack into voxels. Requires mask and pore stacks to 
# have been created. 
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import imageio
from myutils import limit_step

file_in='F:\pd\CT_data\Ren001_Ti64\segmented_test.h5'
#some constants
CT_VOXELSIZE=26.495
BIN_SIZE=600
PPB=math.ceil(BIN_SIZE/CT_VOXELSIZE) 

with h5py.File(file_in, 'r') as f:
    mask_test=np.array(f.get('mask')[10], dtype='uint8')
    height=len(f.get('mask'))
    
porosity_array=np.zeros((math.ceil(height/PPB), math.ceil(mask_test.shape[0]/PPB),
                       math.ceil(mask_test.shape[1]/PPB)), dtype='float64')
mask_array=np.zeros((math.ceil(height/PPB), math.ceil(mask_test.shape[0]/PPB), 
                    math.ceil(mask_test.shape[1]/PPB)), dtype='uint8')
z_i=0
for z_lo, z_hi in limit_step(height, PPB):
    
    with h5py.File('F:\pd\CT_data\Ren001_Ti64\segmented_test.h5', 'r') as f:
        mask=np.array(f.get('mask')[z_lo:z_hi], dtype='uint8')
        pores=np.array(f.get('pores')[z_lo:z_hi], dtype='uint8')
    
    print("Binning layers {0}-{1}".format(z_lo,z_hi))    
    for x_lo,x_hi in limit_step(mask_test.shape[1], PPB):
     
        for y_hi,y_lo in limit_step(mask_test.shape[0], -PPB):
            pore_bin = pores[:, y_lo:y_hi, x_lo:x_hi]
            mask_bin = mask[:, y_lo:y_hi, x_lo:x_hi]
            
            if mask_bin.any()==1:
                porosity_array[z_i, (y_lo//PPB)+1 , x_lo//PPB] = (np.sum(pore_bin==1)/np.sum(mask_bin==1))*100
                #i.e. this is the % porosity of the bin
                mask_array[z_i, (y_lo//PPB)+1 , x_lo//PPB] = 1
    
    z_i=z_i+1
    del(pores, mask)

#porosity_array=porosity_array*1000
#roll and flip
#imageio.volwrite("F:\pd\CT_data\Ren001_Ti64\mask_voxels0-9.tiff", A, format='TIFF')

with h5py.File("F:\\pd\\ML_models\\Ren001_Ti\\600um_voxel\\600um_CT_voxels.h5", 'w') as f:
    f.create_dataset('pores', shape=porosity_array.shape, dtype='float64', data=porosity_array)
    f.create_dataset('mask', shape=mask_array.shape, dtype='uint8', data=mask_array)
#imageio.volwrite("F:\pd\CT_data\Ren001_Ti64\pore_voxels0-9.tiff", porosity_array.astype('uint16'), format='TIFF')
