import numpy as np

def load_system(dump_path, N_steps):
    with open(dump_path, "r") as f:
        
        positions = None
        forces = None
        charges = None
        energies = None

        header_len = 9
        current_line = 0
        N_atoms = np.inf

        atom_data = []
        for line in f: #read file this way to avoid loading it all in memory
            if current_line == 3:
                N_atoms = int(line)
                positions = np.zeros((N_atoms, 3, N_steps))
                forces = np.zeros((N_atoms, 3, N_steps))
                charges = np.zeros(N_atoms)
                energies = np.zeros((N_atoms, N_steps))
                masses = np.zeros(N_atoms)
            elif current_line == 5:
                x_min, x_max = [float(x) for x in line.strip().split()]
            elif current_line == 6:
                y_min, y_max = [float(y) for y in line.strip().split()]
            elif current_line == 7:
                z_min, z_max = [float(z) for z in line.strip().split()]
            elif current_line == 8:
                fields = line.strip().split()[2:]
            elif 9 <= current_line < N_atoms + header_len:
                atom_data.append([float(e) for e in line.strip().split()])
            elif current_line > N_atoms + header_len:
                break
            current_line += 1

        
        atom_data = np.array(atom_data)
        

    with open(dump_path, "r") as f:
        for i in range(N_steps):
            for k in range(header_len): f.readline()

            for j in range(N_atoms):
                line = np.array(f.readline().strip().split())
                # print(np.array(line[3:6]).astype(np.float64))
                positions[j,:,i] = np.array(line[3:6]).astype(np.float64)
                forces[j,:,i] = np.array(line[6:9]).astype(np.float64)
                charges[j] = np.array(line[1]).astype(np.float64)
                energies[j,i] = float(line[9])
                masses[j] = float(line[2])
    
    return positions, forces, energies, masses, charges, np.array([x_max, y_max, z_max])

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

def LJ(r, ep, sig):
    k = (sig/r)**6
    U = 4*ep*((k**2) - k)
    F = 4*ep*((12*(sig**12)/(r**13)) - (6*(sig**6)/(r**7)))
    return U, F

def LJ_NaNa(r):
    ep = 0.1
    sig = 2.583
    return LJ(r, ep, sig)

def LJ_ClCl(r):
    ep = 0.1
    sig = 4.401
    return LJ(r, ep, sig)

def LJ_NaCl(r):
    ep = 0.1
    sig = 3.492
    return LJ(r, ep, sig)

#To calculate LJ part of interaction
def lj_energy_loop(positions, charges, box_sizes, r_cut_real):
    N_atoms = len(charges)

    forces = np.zeros((N_atoms, 3))
    U = np.zeros(N_atoms)

    for i in range(N_atoms):
        for j in range(i+1, N_atoms):
            r_ij = positions[i] - positions[j]
            r_ij = nearest_mirror(r_ij, box_sizes)

            dist_ij = np.linalg.norm(r_ij)

            if dist_ij < r_cut_real:

                if charges[i] == 1.0 and charges[j] == 1.0: #both Na
                    U_ij, F_ij = LJ_NaNa(dist_ij)
                elif charges[i] == -1 and charges[j] == -1: #both Cl
                    U_ij, F_ij = LJ_ClCl(dist_ij)
                else: #Na + Cl
                    U_ij, F_ij = LJ_NaCl(dist_ij)

                r_hat = r_ij / dist_ij 
                F_ij = F_ij*r_hat

                forces[i,:] += F_ij
                forces[j,:] -= F_ij
                U[i] += U_ij

    return U, forces