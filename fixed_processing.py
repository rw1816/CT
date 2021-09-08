from scipy.ndimage import rotate as imrotate3
import matplotlib.pyplot as plt
import cv2
import numpy as np

def rotate_stack(vol, dim, angle=0):
    if angle:
        if dim.lower()=="z":
            vol=imrotate3(vol, -5, axes=(1,2), reshape=True)
        elif dim.lower()=="x":
            vol=imrotate3(vol, -5, axes=(0,2), reshape=True)
        elif dim.lower()=="y":
            vol=imrotate3(vol, -5, axes=(0,1), reshape=True)
        else:
            'Error! invalid rotation plane specified'
    else:
        running=True
        while running:
            plt.imshow(vol[10])
            plt.show()
            try_ang=float(input("Enter relative rotation angle -- pass nothing to exit"))
            if dim.lower()=="z":
                vol=imrotate3(vol, try_ang, axes=(1,2), reshape=True)
            elif dim.lower()=="x":
                vol=imrotate3(vol, try_ang, axes=(0,2), reshape=True)
            elif dim.lower()=="y":
                vol=imrotate3(vol, try_ang, axes=(0,1), reshape=True)
            else:
                'Error! invalid rotation plane specified'
            
            if try_ang==0:
                running=False
                          
    return vol

def cropNalign(vol, thrs=5000):
    ret, thres1 = cv2.threshold(vol[5], 2000, 1, cv2.THRESH_BINARY)
    maxs=np.max(np.where(thres1==1), axis=1)
    mins=np.min(np.where(thres1==1), axis=1)
    vol=vol[:,mins[0]-1:maxs[0]+1, mins[1]-1:maxs[1]+1]
    
    return vol, (maxs, mins)

def crop(vol, ids):
    vol=vol[:,ids[1][0]-1:ids[0][0]+1, ids[1][1]-1:ids[0][1]+1]
    return vol