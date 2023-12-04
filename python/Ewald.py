import numpy as np
import scipy.special as ss
from helper import *
import os

def direct_sum(r, q, alpha, real_lat, r_cut):
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
                            U_direct[i] += q[i] * q[j] * ss.erfc(alpha*dist_ijn/np.sqrt(2)) / (dist_ijn)

                            F_ij = q[i] * q[j] #TODO  #TODO

                            r_hat = r_ijn / dist_ijn 
                            F_ij = F_ij*r_hat

                            F_direct[i,:] += F_ij


    return 0.5*U_direct, F_direct


def abs2(x : np.complex128):
    return (x.real**2) + (x.imag**2)

def rec_sum(r, q, alpha, recip_lat, k_cut, V):
    N_atoms = len(q)
    U_rec = np.zeros(N_atoms)
    F_rec = np.zeros((N_atoms,3))

    Kx, Ky, Kz = k_cut
    a_inv_sq = (1/alpha)**2

    #Precompute k-vectors and structure factor
    # k_vecs = np.zeros((2*Kx+1, 2*Ky+1, 2*Kz+1, 3))
    # k_sqs = np.zeros((2*Kx+1, 2*Ky+1, 2*Kz+1))
    # S = np.zeros((2*Kx+1, 2*Ky+1, 2*Kz+1))

    # for k1 in range(-Kx,Kx+1):
    #     for k2 in range(-Ky,Ky+1):
    #         for k3 in range(-Kz,Kz+1):
    #             # k_vecs[k1 + Kx, k2 + Ky, k3 + Kz , :] = k1*recip_lat[0,:] + k2*recip_lat[1,:] + k3*recip_lat[2,:]
    #             k_vec = k1*recip_lat[0,:] + k2*recip_lat[1,:] + k3*recip_lat[2,:]
    #             k_sqs[k1 + Kx, k2 + Ky, k3 + Kz] = np.dot(k_vec, k_vec)
    #             S[k1 + Kx, k2 + Ky, k3 + Kz] = abs2(np.sum([q*np.exp(1j*(np.dot(k_vec, r[i,:]))) for i in range(N_atoms)]))

    #Calculate energy
    # for i in range(N_atoms):
    #     for j in range(N_atoms):

            # r_ij = r[i] - r[j]

    for k1 in range(-Kx,Kx+1):
        for k2 in range(-Ky,Ky+1):
            for k3 in range(-Kz,Kz+1):

                if k1 == 0 and k2 == 0 and k3 == 0:
                        continue
                
                k_vec = k1*recip_lat[0,:] + k2*recip_lat[1,:] + k3*recip_lat[2,:]
                k_sq = np.dot(k_vec, k_vec)
                S_sq = abs2(np.sum([q[i]*np.exp(1j*(np.dot(k_vec, r[i,:]))) for i in range(N_atoms)]))

                # U_rec[i] += np.real(np.exp(-(p_a*k_sq) + two_pi*1j*np.dot(k_vec, r_ij))/k_sq)
                U_rec[i] += (np.exp(-a_inv_sq*k_sq/2)/k_sq)*S_sq
                # F_rec[i] += 0.0 #TODO

    # U_rec[i] *= q[i]*q[j]
    
    return U_rec/(2*V), F_rec

def self_energy(q, alpha):
    return (-alpha/np.sqrt(2*np.pi))*np.sum(np.square(q))

#http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#coulomb-interaction-with-ewald-summation
def get_k_cut(delta, alpha, L):
    #Find smallest k_max in each dim that gives error < delta

    error = lambda k_max,d : (k_max*np.sqrt(d*alpha)/20)*np.exp(-((np.pi*k_max/(d*alpha))**2))

    kmaxX = 1; kmaxY = 1; kmaxZ = 1;
    while error(kmaxX, L[0]) > delta:
        kmaxX += 1
    
    while error(kmaxY, L[1]) > delta:
        kmaxY += 1

    while error(kmaxZ, L[2]) > delta:
        kmaxZ += 1
    
    return kmaxX, kmaxY, kmaxZ


def Ewald(r, q, real_lat_vec, error_tol, r_cut_real):

    L = np.array([np.linalg.norm(real_lat_vec[i,:]) for i in range(real_lat_vec.shape[0])])
    # OpenMM parameterizes like this:
    # http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#coulomb-interaction-with-particle-mesh-ewald
    alpha = np.sqrt(-np.log(2*error_tol))/r_cut_real
    #Look at FFT implementation it can be better if this is a factor of some prime number
    k_cut = get_k_cut(error_tol, alpha, L)

    V = vol(real_lat_vec)
    recip_lat = reciprocal_vecs(real_lat_vec)
    
    
    E_dir, F_dir = direct_sum(r, q, alpha, real_lat_vec, r_cut_real)
    print(f"\tDirect Energy Calculated")
    print(f"\tN-Mesh {k_cut}")
    E_rec, F_rec = rec_sum(r, q, alpha, recip_lat, k_cut, V)
    print(f"\tRecip Energy Calculated")
    self_eng = self_energy(q, alpha)

    A = 332.218 #1/4pie_0 in correct units

    print(f"Real Energy: {np.sum(E_dir)*A}")
    print(f"Recip Energy {np.sum(E_rec)*A*4*np.pi}")
    print(f"Self Energy {np.sum(self_eng)*A}")
    print(f"Total Ewald Eng {(np.sum(E_dir) + 4*np.pi*np.sum(E_rec) + np.sum(self_eng))*A}")

    U_tot = A*(E_dir + 4*np.pi*E_rec + self_eng)
    F_tot = F_dir + F_rec #*need to convert these units too
    return U_tot, F_tot


if __name__ == "__main__":

    r_cut_lj = 7.0 #needs to be less then 8 for box size w/ 3 UC
    r_cut_real = 10.0 #kinda picked randomly
    error_tol = 1e-4 

    dump_path = os.path.join(r"../test_data\salt_sim\dump.atom")


    lattice_param = 5.62 #Angstroms
    N_uc = 3
    real_lat_vecs = np.array([[N_uc*lattice_param,0,0],[0,N_uc*lattice_param,0],[0,0,N_uc*lattice_param]]) #often denoted a
    recip_lat_vecs = reciprocal_vecs(real_lat_vecs)

    #positions are (Nx3) masses, charges (Nx1), boxsize (3x1)
    N_steps = 11
    positions, forces_lmp, eng_lmp, masses, charges, box_sizes = load_system(dump_path, N_steps)
    for i in range(1):
        print(f"ON LOOP ITERATION {i}")
        U_LJ, F_LJ = lj_energy_loop(positions[:,:,i], charges, box_sizes, r_cut_lj)

        print(f"\t LJ Energy Calculated")
        #TBH no clue how to use this
        # i_inds, j_inds, _ = cl.neighborlist(positions[:,:,i], r_cut_neighbor, unitcell=np.array([1, 1, 1]))
        U_ewald, F_ewald = Ewald(positions[:,:,i], charges, real_lat_vecs, error_tol, r_cut_real)
        # print(f"\t PME Energy Calculated")


