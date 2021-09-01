import numpy as np
from skimage import measure
import h5py
import visvis as vv
from myutils import limit_step

data_path='F:\pd\CT_data\Ren001_Ti64\segmented_test.h5'
with h5py.File(data_path, 'r') as f:
        length=len(f.get('pores'))
        length2=len(f.get('mask'))
## we must block through the scan due to limited RAM
sum_pores=0
for start,end in limit_step(length, 500):
    with h5py.File(data_path, 'r') as f:
        pores=np.array(f.get('pores')[start:end], dtype='uint8')
    sum_pores=sum_pores+np.sum(pores==1)
    #verts, faces, normals , _ = measure.marching_cubes_lewiner(pores)
    #mesh = vv.mesh(np.fliplr(verts), faces, normals)
    #vv.meshWrite('F:\pd\CT_data\Ren001_Ti64\Renders\REN001_poreMesh{0}-{1}.stl'.format(start,end),
                 #mesh)
sum_mask=0
for start,end in limit_step(length, 500):
    with h5py.File(data_path, 'r') as f:
        mask=np.array(f.get('mask')[start:end], dtype='uint8')
    sum_mask=sum_mask+np.sum(mask==1)
    #verts, faces, normals , _ = measure.marching_cubes_lewiner(pores, step_size=3)
    #mesh = vv.mesh(np.fliplr(verts), faces, normals)
    #vv.meshWrite('F:\pd\CT_data\Ren001_Ti64\Renders\REN001_maskMesh{0}-{1}.stl'.format(start,end),
                 #mesh)

tot_porosity=(sum_pores/sum_mask)*100
# display mesh, optional
#app = vv.use()
#app.Run()


