# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:47:31 2019

individual pore extractions

@author: rw1816
"""
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.draw import ellipsoid
import h5py
from myutils import limit_step

filename=data_path='F:\pd\CT_data\Ren001_Ti64\segmented_test.h5'
centroid=np.zeros((3))
poreArea = []
poreEccentricity = []
poreMajor_axis_length = []
poreMinor_axis_length = []
porePerimeter = []
poreOrientation = []
bbox = np.zeros((6))
poreAR = []
with h5py.File(filename, 'r') as f:
    length=len(f.get('pores'))
    
for start,end in limit_step(length, 100):
    with h5py.File(filename, 'r') as f:
        pores=np.array(f.get('pores')[start:end])
        print("{0} loaded".format(end))
        label_img = measure.label(pores)
        res = measure.regionprops(label_img)
        
    for props in res:
        if props.area > 5: #filter out anything less than 5 voxels in size
            cent_raw=props.centroid
            cent_raw=(cent_raw[0]+start, cent_raw[1],cent_raw[2])
            centroid= np.vstack((centroid,cent_raw))
            poreArea = np.append(poreArea,props.area)
            bbox =  np.vstack((bbox , props.bbox))
            #poreEccentricity= np.append(poreArea,props.eccentricity)
            poreMajor_axis_length = np.append(poreMajor_axis_length,props.major_axis_length)
            poreMinor_axis_length = np.append(poreMinor_axis_length,props.minor_axis_length)
            poreAR = np.append(poreAR, props.minor_axis_length/ props.major_axis_length)
            #porePerimeter = np.append(poreArea,props.perimeter)
            #poreOrientation = np.append(poreArea,props.orientation)

centroid=centroid[1:]
bbox=bbox[1:]
#%%
# initialise some stuff
#colours = ['r','g','b']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.size'] = '20.0'
matplotlib.rcParams['text.usetex'] = False
voxel_size = 6.7
nBins = 15
#%% 
#size distribution plot
plt.close('all')
poreVol = poreArea*(voxel_size**3)
pore_length = poreMajor_axis_length*(voxel_size)
normLength = np.cbrt(poreVol)
f=plt.figure(figsize=(4, 4), facecolor='w')
ax=plt.axes() 
n, bins, patches = plt.hist(normLength, bins=(np.arange(0, 360, 10)), histtype='bar', ec='black', linewidth=0.5)
ax.set(xlabel = r'Normalised cubic length $\mu$m', ylabel = 'Counts', yscale='log')
plt.savefig('plots\size_distribution.pdf', facecolor='w', bbox_inches='tight')

f=plt.figure(figsize=(4, 4), facecolor='w')
ax=plt.axes() 
n, bins, patches = plt.hist(pore_length, bins=(np.arange(0, 2000, 50)), histtype='bar', ec='black', linewidth=0.5)
ax.set(xlabel = r'Major axis length $\mu$m', ylabel = 'Counts', yscale='log')
plt.savefig('plots\length_distribution.pdf', facecolor='w', bbox_inches='tight')

#%%
# binning and sorting the data
plt.close('all')
pore_length = poreMajor_axis_length*(voxel_size)

hist, bin_edges = np.histogram(poreX*voxel_size, bins = nBins) #x co-ord here is z in build, this command bins all
# pores by their x co-ordinate
bin_num = np.digitize(poreX*voxel_size, bins = bin_edges) #assigns each pore in the regionprops vector
# its bin number according to above calulated bin edges
bin_width = bin_edges[4]-bin_edges[3]
bin_centre = bin_edges + bin_width/2
bin_centre = bin_centre[0:-1]
binned_vol = np.array(np.zeros(nBins)) #  only preallocation
binned_length = np.array(np.zeros(nBins))
max_size = np.array(np.zeros(nBins))
max_length = np.array(np.zeros(nBins))
aspect = np.array(np.zeros(nBins))

for i in range(0, nBins): 
    binned_vol[i] = np.sum(normLength[bin_num==i+1]) #this gives the TOTAL PORE VOLUME in each bin
    binned_length[i] = np.sum(pore_length[bin_num==i+1]) #this gives the TOTAL PORE LENGTH in each bin
    max_size[i] = max(normLength[bin_num==i+1])
    max_length[i] = max(pore_length[bin_num==i+1])
    aspect[i] = np.average(poreAR[bin_num==i+1])
    
#%% plotting     
    
av_vol=binned_vol/hist    
av_length=binned_length/hist 
    
  
f=plt.figure(figsize=(4, 4), facecolor='w')
ax=plt.axes() 
ax.set(xlabel = '$z$ position in notch $(mm)$', ylabel = r'Normalised cubic length $\mu$m', yscale='log')
plt.bar(bin_centre, binned_vol, width=bin_width, align='center', ec='black')
plt.savefig('plots\sizeVsZ.pdf', facecolor='w', bbox_inches='tight')

f=plt.figure(figsize=(4, 4), facecolor='w')
ax=plt.axes()
ax.set(xlabel = '$z$ position in notch $(mm)$', ylabel = 'Number of pores') 
plt.bar(bin_centre, hist, width=bin_width, align='center', ec='black')
plt.savefig(r'plots\numVsZ.pdf', facecolor='w', bbox_inches='tight')

plt.plot(bin_centre, max_size)
ax=plt.gca()
ax.set(xlabel = '$z$ position in notch $(mm)$', ylabel = r'Maximum size/ normalised cubic length $\mu$m')
plt.savefig(r'plots\max_sizeVsZ.pdf', facecolor='w', bbox_inches='tight')

plt.plot(bin_centre, max_length)
ax=plt.gca()
ax.set(xlabel = '$z$ position in notch $(mm)$', ylabel = r'Maximum major axis length length $\mu$m')
plt.savefig(r'plots\maxMajAxVsZ.pdf', facecolor='w', bbox_inches='tight')

#%% aspect ratio plots

#%% size distribution vs contribution to overall porosity quantity
hist_size, bin_edges_size = np.histogram(normLength, bins = len((np.arange(0, 360, 10))))
bin_num_size = np.digitize(normLength, bins = bin_edges_size)
binned_poreVol = np.array(np.zeros(len(np.arange(0, 360, 10))))

for i in range(0, len(np.arange(0, 360, 10))): 
    binned_poreVol[i] = np.sum(poreVol[bin_num_size==i+1]) #this gives the TOTAL PORE VOLUME in each size bin
    
total_poreVol = np.sum(poreVol)
binned_pp = (binned_poreVol/ total_poreVol)*100
f=plt.figure(figsize=(4, 4), facecolor='w')
ax=plt.axes()
ax.set(xlabel = r'Normalised length ($\mu$m)', ylabel = 'Percentage of total porosity') 
plt.bar((np.arange(0, 360, 10)), binned_pp, width=10, align='center', ec='black')
plt.savefig(r'plots\pore_perc_vs_normLen.pdf', facecolor='w', bbox_inches='tight')

#%% finding the largest and longest pores
import pyvista as pv
from skimage import measure

largest_ind = np.where(poreArea == max(poreArea))
longest_ind = np.where(poreArea == max(poreArea))
largest_bb = res[largest_ind[0][0]].bbox
largest_centroid = res[largest_ind[0][0]].centroid
longest_bb = res[largest_ind[0][0]].bbox

verts_long, faces_long, normals_long, values_long = measure.marching_cubes_lewiner(V
   [longest_bb[0]-1:longest_bb[3]+1, longest_bb[1]-1:longest_bb[4]+1, longest_bb[2]-1:longest_bb[5]+1])
faces_ind = np.zeros((len(faces_long), 4))
faces_ind[:,0]=3
faces_ind[:,1:4]=faces_long  #padding out the faces array to include a column of 3s stating they are triangular
surf_long = pv.PolyData(verts_long, faces_ind)
surf_long = surf_long.smooth(n_iter=1000)

verts_large, faces_large, normals_large, values_large = measure.marching_cubes_lewiner(M)
faces_large_new = np.zeros((len(faces_large), 4))
faces_large_new[:,0]=3
faces_large_new[:,1:4]=faces_large
surf_large = pv.PolyData(verts_large, faces_large_new)


p = pv.BackgroundPlotter()
p.add_mesh(surf_long, color='Red', smooth_shading=True)
p.background_color = 'white'

#p1 = pv.BackgroundPlotter()
#p1.add_mesh(surf_large, color='Red', smooth_shading=True)
#p.background_color = 'white'

