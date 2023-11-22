import numpy as np

def load_system(dump_path : str):
    #Just parse lammps dump file that has x y z charge mass info
    pass

#same idea as: https://github.com/Jonas-Finkler/ewald-summation/blob/6c399ba021184c1ca66540642ab0b0d8f236cce3/src/atoms/pointsets.f90#L121
#returns the number of cells in the direction of each lattice vector that are needed to find all neighbors within cutoff
def get_N_cells(lattice_vecs, r_cut):
    axb = np.cross(lattice_vecs[:, 0], lattice_vecs[:, 1])
    axb /= np.linalg.norm(axb)
    
    axc = np.cross(lattice_vecs[:, 0], lattice_vecs[:, 2])
    axc /= np.linalg.norm(axc)
    
    bxc = np.cross(lattice_vecs[:, 1], lattice_vecs[:, 2])
    bxc /= np.linalg.norm(bxc)

    proj = np.zeros(3)
    proj[0] = np.dot(lattice_vecs[:, 0], bxc)
    proj[1] = np.dot(lattice_vecs[:, 1], axc)
    proj[2] = np.dot(lattice_vecs[:, 2], axb)

    n = np.ceil(r_cut / np.abs(proj))

    return n.astype(int)

def reciprocal_vecs(lat_vecs):
    V = np.dot(lat_vecs[:,0],np.cross(lat_vecs[:,1], lat_vecs[:,2]))
    m1 = np.cross(lat_vecs[:,1], lat_vecs[:,2])/V
    m2 = np.cross(lat_vecs[:,2], lat_vecs[:,0])/V
    m3 = np.cross(lat_vecs[:,0], lat_vecs[:,1])/V
    return np.array([m1,m2,m3])