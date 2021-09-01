#main script
#read in some of the scan
import numpy as np
import os
from fixed_processing import rotate_stack, cropNalign, crop
from CT.v2_1.segmentation_routines import mask_and_pores, isolate_pores, create_img_dict
import h5py
import mpi4py
from myutils import parallel_chunker
import time
import matplotlib.pyplot as plt
import pickle

tic=time.time()
root_dir=os.path.normpath('F:\pd\CT_data\Ren001_Ti64\Merge attempt 2')
scan_name="Merge2.raw"
file_path=os.path.join(root_dir, scan_name)

img_params=create_img_dict(os.path.join(root_dir, (scan_name[:-4]+'_imgParams.pickle')))
#N.B. z axis is 0 with memmap
RAW = np.memmap(file_path, dtype='uint16', mode='r', shape=(2911,1721,2070))
#%%
from CT.v2_1.segmentation_routines import mask_and_pores, isolate_pores, create_img_dict
write_path='F:\pd\CT_data\Ren001_Ti64\segmented_test.h5'
tic=time.time()
with h5py.File(write_path, 'w', rdcc_nbytes=8e9) as f:
    d_pores=f.create_dataset("pores", (1,1,1), maxshape=(RAW.shape[0],2000,2500), 
                chunks=(1, 2000, 2500), dtype='uint8')
    d_mask=f.create_dataset("mask", (1,1,1), maxshape=(RAW.shape[0],2000,2500), 
                chunks=(1, 2000, 2500), dtype='uint8')
    
    FIRST_LOOP=True
    for start, end in matrix_chunker(100, RAW, offset=20):
        
        raw=RAW[start:end,:,:]
        raw=rotate_stack(raw, 'z', angle=-5)
        #raw=rotate_stack(raw, 'y', angle=-5)
        if FIRST_LOOP:
            raw, ids =cropNalign(raw, thrs=5000)
            d_pores.resize((RAW.shape[0],raw.shape[1],raw.shape[2]))
            d_mask.resize((RAW.shape[0],raw.shape[1],raw.shape[2]))
        else:
            raw=crop(raw, ids)
        mask=np.zeros(raw.shape, dtype='uint8')
        pores=np.zeros(raw.shape, dtype='uint8')
        for i, img in enumerate(raw):
        
            mask[i], pores[i]=mask_and_pores(img, img_params, adapt=True, gauss=True)
        d_pores[start:end,:,:]=pores[:]
        d_mask[start:end,::]=mask[:]
        del(raw)
        print('Completed {0}'.format(end))
        FIRST_LOOP=False
        
toc=time.time()
print('Total time taken = {0}s'.format(toc-tic))

#%%
#update dict
old_params=create_img_dict(os.path.join(root_dir, (scan_name[:-4]+'_imgParams.pickle')))
if old_params!=img_params:
    print('Parameters changed, overwriting...')
    with open(os.path.join(root_dir, (scan_name[:-4]+'_imgParams.pickle')), 'wb') as p:
        pickle.dump(img_params, p)