import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

def checkStack(seg_fn, raw_fn, raw_shape, i, offset):
    """
    Function or script to check the success, or quality, of the mask/pore segmentation, 
    as process by segmentStack.py. Will plot the original image, the mask of the 
    part and the segmented out pore stack side by side.
    
    Inputs: filename = path to file to check (output .h5 from segmentStack)
            i = index & range of imgs in the stack to compare
            offset = number of dropped image slices from the original stack
                    (eg. where we skipped blank space at start, supports etc.)
            raw_fn, raw_shape = filename and shape of raw stack
            
    Outputs: None, will plot 3 images
    """
    
    with h5py.File(seg_fn, 'r') as f:
        pores=np.array(f.get('pores')[i], dtype='uint8')
        mask=np.array(f.get('mask')[i], dtype='uint8')
    
    RAW=np.memmap(raw_fn, dtype='uint16', mode='r', shape=raw_shape)    
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(mask)
    ax[1].imshow(pores)
    ax[2].imshow(RAW[i+offset])

    return 

if __name__ == "__main__":
    
    root_dir=os.path.normpath('F:\pd\CT_data\Ren001_Ti64\Merge attempt 2')
    scan_name="Merge2.raw"
    raw_path=os.path.join(root_dir, scan_name)
    
    checkStack("F:\\pd\\CT_data\\Ren001_Ti64\\segmented.h5", raw_path, (2911,1721,2070),
               0, 20)
    