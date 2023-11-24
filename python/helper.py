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

def vol(lat_vecs):
    return np.dot(lat_vecs[:,0],np.cross(lat_vecs[:,1], lat_vecs[:,2]))

def reciprocal_vecs(lat_vecs):
    V = vol(lat_vecs)
    m1 = np.cross(lat_vecs[:,1], lat_vecs[:,2])/V
    m2 = np.cross(lat_vecs[:,2], lat_vecs[:,0])/V
    m3 = np.cross(lat_vecs[:,0], lat_vecs[:,1])/V
    return np.array([m1,m2,m3])

def nearest_mirror(r_ij,box_sizes):
    Lx,Ly,Lz = box_sizes
    r_x,r_y,r_z = r_ij
        
    if r_x > Lx/2:
        r_x -= Lx
    elif r_x < -Lx/2:
        r_x += Lx
        
    if r_y > Ly/2:
        r_y -= Ly
    elif r_y < -Ly/2:
        r_y += Ly 
        
    if r_z > Lz/2:
        r_z -= Lz
    elif r_z < -Lz/2:
        r_z += Lz

    return np.array([r_x,r_y,r_z])

#To calculate LJ part of interaction
def lj_energy_loop(positions, charges, box_sizes, r_cut_real):
    N_atoms = len(charges)

    forces = np.zeros(N_atoms, 3)
    U = np.zeros(N_atoms)

    for i in range(N_atoms):
        for j in range(i+1, N_atoms):
            r_ij = positions[i] - positions[j]
            r_ij = nearest_mirror(r_ij, box_sizes)

            dist_ij = np.norm(r_ij)

            if dist_ij < r_cut_real:

                if charges[i] == 1.0 and charges[j] == 1.0: #both Na
                    U_ij, F_ij = None, None
                elif charges[i] == -1 and charges[j] == -1 #both Cl
                    U_ij, F_ij = None, None
                else: #Na + Cl
                    U_ij, F_ij = None, None

                r_hat = r_ij / dist_ij 
                F_ij = F_ij*r_hat

                forces[i,:] += F_ij
                forces[j,:] -= F_ij
                U[i] += U_ij