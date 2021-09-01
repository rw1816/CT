"""
Script to compare the shape of each layer of the voxel array of the part, produced
from both the X-ray CT scan and the monitoring data. There should be the same configuration
of voxels for fitting and comparison.

"""
import os.path
import imageio
import numpy as np

CT_file=os.path.normpath("F:\pd\CT_data\Ren001_Ti64\mask_voxels.tiff")
monitoring_file=os.path.normpath("F:\pd\monitoring_data\Ren001\Ren001_PD1voxels_0.6_2.tiff")

monitored_voxels=imageio.volread(monitoring_file)
CT_voxels=imageio.volread(CT_file)

monitored_shape=np.zeros((len(monitored_voxels)))
CT_shape=np.zeros((len(CT_voxels)))

for i, frame in enumerate(CT_voxels):
    CT_shape[i]=np.size(frame[frame>0])
    
for i, frame in enumerate(monitored_voxels):
    monitored_shape[i]=np.size(frame[frame>1000])