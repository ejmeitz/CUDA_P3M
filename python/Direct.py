import numpy as np
import scipy.special as ss
from helper import *
import os

def direct_sum(r, q, real_lat, r_cut):
    N_atoms = len(q)
    U_direct = np.zeros(N_atoms)
    F_direct = np.zeros((N_atoms,3))

    #Calculate number of cells in each direction that could be reached given r_cut
    #* Can be pre-calculated outside of function
    N1, N2, N3 = get_N_cells(real_lat, r_cut)
    print(f"N-Real [{N1} {N2} {N3}]")

    #* Fix this so it interacts with its own mirror particles (just not itself)
    for n1 in range(-N1,N1+1):
        for n2 in range(-N2,N2+1):
            for n3 in range(-N3,N3+1):

                n_vec = n1*real_lat[0,:] + n2*real_lat[1,:] + n3*real_lat[2,:]
                
                #How tf u use neighbor lists here
                for i in range(N_atoms):
                    for j in range(N_atoms): #think u can only do i < j if n_vec = 0,0,0, check once it works
                        
                        #Only exclude self interaction in base unit cell
                        if n1 == 0 and n2 == 0 and n3 == 0 and i == j:
                            continue


                        r_ijn = r[i] - r[j] + n_vec
                        dist_ijn = np.linalg.norm(r_ijn)

                        if dist_ijn < r_cut:
                            U_direct[i] += q[i] * q[j] / dist_ijn

                            F_ij = q[i] * q[j] #TODO  #TODO

                            r_hat = r_ijn / dist_ijn 
                            F_ij = F_ij*r_hat

                            F_direct[i,:] += F_ij


    return 0.5*U_direct, F_direct

def Direct(r, q, real_lat_vec, r_cut_real):

    L = np.array([np.linalg.norm(real_lat_vec[i,:]) for i in range(real_lat_vec.shape[0])])

    
    E_dir, F_dir = direct_sum(r, q, real_lat_vec, r_cut_real)


    A = 332.0637128 #1/4pie_0 in correct units given unit charges and angstroms to get kcal/mol energies

    print(f"Real Energy: {np.sum(E_dir)*A}")


    return E_dir*A, F_dir


if __name__ == "__main__":

    r_cut_lj = 7.0 #needs to be less then 8 for box size w/ 3 UC
    r_cut_real = 10.0 #does this need to be less than L/2????

    dump_path = os.path.join(r"../test_data\salt_sim\dump.atom")

    lattice_param = 5.62 #Angstroms
    N_uc = 3
    real_lat_vecs = np.array([[N_uc*lattice_param,0,0],[0,N_uc*lattice_param,0],[0,0,N_uc*lattice_param]]) #often denoted a

    #positions are (Nx3) masses, charges (Nx1), boxsize (3x1)
    N_steps = 11
    positions, forces_lmp, eng_lmp, masses, charges, box_sizes = load_system(dump_path, N_steps)
    for i in range(2):
        print(f"ON LOOP ITERATION {i}")
        U_LJ, F_LJ = lj_energy_loop(positions[:,:,i], charges, box_sizes, r_cut_lj)

        print(f"\t LJ Energy Calculated")
        U_ewald, F_ewald = Direct(positions[:,:,i], charges, real_lat_vecs, r_cut_real)

