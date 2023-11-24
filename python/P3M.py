import CellListMap as cl #neighbor list builder from Julia: https://m3g.github.io/CellListMap.jl/stable/python/
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
import scipy.signal as sig
import math

from .helper import *

#similar code: https://github.com/jht0664/structurefactor_spme/blob/master/run_sq.py

#Calculate E_direct
def particle_particle(r, q, alpha, r_cut, real_lat):
    N_atoms = len(q)
    U_direct = np.zeros(N_atoms)
    F_direct = np.zeros(N_atoms)


    #Calculate number of cells in each direction that could be reached given r_cut
    #* Can be pre-calculated outside of function
    N1, N2, N3 = get_N_cells(real_lat, r_cut)


    #* Fix this so it interacts with its own mirror particles (just not itself)
    for n1 in range(-N1,N1+1):
        for n2 in range(-N2,N2+1):
            for n3 in range(-N3,N3+1):

                n_vec = np.array(n1,n2,n3)
                
                #How tf u use neighbor lists here
                for i in range(N_atoms):
                    for j in range(i+1, N_atoms): #they had from 1-N and divide by 2 at end, check that its the same

                        r_ijn = r[j] - r[i] + n_vec
                        dist_ijn = np.norm(r_ijn)

                        if dist_ijn < r_cut:
                            U_direct[i] += q[i] * q[j] * ss.erfc(alpha*dist_ijn) / dist_ijn
                            F_ij = q[i] * q[j] * something #TODO

                            F_direct[i] += F_ij
                            F_direct[j] -= F_ij #on GPU this is not worth doing

    
    return U_direct, F_direct

#From appendix B
def M(u, n):
    if n > 2:
        return (u/(n-1))*M(u,n-1) + ((n-u)/(n-1))*M(u-1,n-1)
    else:
        return 1 - np.abs(u-1)

#equivalent, time to see whats faster
# def M(u, n):
#     return (1/math.factorial(n-1)) * np.sum([((-1)**k)*math.comb(n,k)*np.power(max(u-k, 0), n-1) for k in range(n+1)])

def dMdu(u,n):
    return M(u, n-1) - M(u-1, n-1)

# Make this non-allocating
# Make this not use np.exp(j) just manage cos() and sin() manually
def b(m, K, n): #n is spline order
    return np.exp(2*np.pi*1.0j*(n-1)*m/K) / sum([M(k+1, n)*np.exp(2*np.pi*1.0j*m*k/K) for k in range(n-1)])

def calc_B(m1,m2,m3,K1,K2,K3,n):
    b1 = b(m1,K1,n)
    b2 = b(m2,K2,n)
    b3 = b(m3,K3,n)
    return (b1.real**2 + b1.imag**2)*(b2.real**2 + b2.imag**2)*(b3.real**2 + b3.imag**2)


def calc_C(alpha, V, m1, m2, m3, K1, K2, K3):
    if (m1 == 0 and m2 == 0 and m3 == 0):
        return 0
    else:
        if m1 <= 0 or m1 <= K1/2:
            m1 -= K1
        elif m2 <= 0 or m2 <= K2/2:
            m2 -= K2
        elif m3 <= 0 or m3 <= K3/2:
            m3 -= K3
        m_sq = m1*m1 + m2*m2 + m3*m3 #is this what the paper means?
        return (1/(np.pi*V))*(np.exp(-(np.pi**2)*m_sq/(alpha**2))/m_sq)
    

def calc_BC(alpha, V, K1, K2, K3, n):
    BC = np.zeros((K1,K2,K3))

    for m1 in range(K1):
        for m2 in range(K2):
            for m3 in range(K3):
                BC = calc_B(m1, m2, m3, K1, K2, K3, n) * calc_C(alpha, V, m1, m2, m3, K1, K2, K3)

    return BC


#Calculate E_reciprocal
def particle_mesh(r, q, real_lat, alpha, spline_interp_order, k_max, mesh_dims):

    N_atoms = len(q)

    recip_lat = reciprocal_vecs(real_lat)
    
    K1, K2, K3 = mesh_dims

    V = vol(real_lat)

    #convert coordinates to fractional
    #make this non-allocating in Julia
    #* Assumes rows of recip lat are each vector
    u = np.array([mesh_dims * np.matmul(recip_lat, r[i,:]) for i in range(r.shape[0])])

    #Fill Q array (interpolate charge onto grid)
    Q = np.zeros((K1, K2, K3))
    dQdr = np.zeros((N_atoms, 3, K1, K2, K3)) #deriv in real space
    BC = calc_BC(alpha, V, K1, K2, K3, spline_interp_order)

    for k1 in range(K1):
        for k2 in range(K2):
            for k3 in range(K3):

                 
                #* not sure this loop is right
                #& my current interpretation is that for a given grid point (k1,k2,k3) get contribution from all other grid pts
                for p1 in range(K1):
                    for p2 in range(K2):
                        for p3 in range(K3):                            
                            for i in range(N_atoms):
                                u0 = u[i,0] - k1 - p1*K1; u1 = u[i,1]- k2 - p2*K2; u2 = u[i,2] - k3 - p3*K3

                                I = dMdu(u0) * M(u1) * M(u2)
                                II = M(u0) * dMdu(u1) * M(u2)
                                III = M(u0) * M(u1) * dMdu(u2)

                                #dQdr_i0 #& not sure indexing is right on recip_lat
                                dQdr[i, 0, k1, k2, k3] += (K1 * recip_lat[0,0] * I) + (K2 * recip_lat[1,0] * II) + (K3 * recip_lat[2,0]  * III)
                                #dQdr_i1
                                dQdr[i, 1, k1, k2, k3] += (K1 * recip_lat[0,1] * I) + (K2 * recip_lat[1,1] * II) + (K3 * recip_lat[2,1]  * III)
                                #dQdr_i2
                                dQdr[i, 2, k1, k2, k3] += (K1 * recip_lat[0,2] * I) + (K2 * recip_lat[1,2] * II) + (K3 * recip_lat[2,2]  * III)

                                #Real space charge interpolated onto mesh
                                Q[k1,k2,k3] += q[i] * M(u0) * M(u1) * M(u2)
                                

    
    #Invert Q (do in place on GPU??)
    Q_recip = np.fft.fftn(Q)

    E_out = np.zeros((K1,K2,K3)) #how to get this per atom??

    for m1 in range(K1):
        for m2 in range(K2):
            for m3 in range(K3):
                if m1 == 0 and m2 == 0 and m3 == 0:
                    continue

                E_out[m1,m2,m3] += Q_recip[m1,m2,m2]*Q_recip[-m1,-m2,-m3]*BC[m1,m2,m3]

                #Update values in Q array to make conv (theta_rec, Q) easier
                Q_recip[m1,m2,m3] *= BC[m1,m2,m3]


    E_out /= (2*np.pi*V)

    #Q_recipt has B and C multipled into it now, so IFFT of this is conv(Theta_rec, Q)
    theta_rec = np.fft.ifft(Q)

    F_out = np.zeros(N_atoms, 3)
    for i in range(N_atoms):
        for dir in range(3):
            F_out[i, dir] = -np.sum(dQdr[i,dir,:,:,:] * theta_rec) #do this without allocating
 

    return E_out, F_out



def self_energy(q, alpha):
   return -(alpha/np.sqrt(np.pi)) * np.sum(np.square(q))



def PME(r, q, real_lat_vec, error_tol, r_cut_real, spline_interp_order):


    L = [np.norm(real_lat_vec[i,:]) for i in range(real_lat_vec.shape[0])]

    # OpenMM parameterizes like this:
    # http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#coulomb-interaction-with-particle-mesh-ewald
    alpha = np.sqrt(-np.log(2*error_tol))/r_cut_real
    #Look at FFT implementation it can be better if this is a factor of some prime number
    n_mesh = np.ceil(2*alpha*L/(3*np.power(error_tol, 0.2))) #assumes cubic box

    pp_energy, pp_force = particle_particle(r, q, alpha, r_cut_real, real_lat_vec)
    pm_energy, pm_force = particle_mesh(r, q, real_lat_vec, alpha, spline_interp_order, n_mesh)
    self_eng = self_energy(r, q)

    U_SPME = pp_energy + pm_energy + self_eng
    F_SPME = pp_force + pm_force

    # This artifact can be avoided by removing the average net force
    # from each atom at each step of the simulation

    #TODO
    avg_net_force = None #tf u take the average of?


    return U_SPME, F_SPME - avg_net_force


def brute_force_energy(r, charges):
    #They do the same sums as before but keep calcluating until change is less than machine eps
    pass

def rms_error(a, b):
    return np.sqrt(np.sum(np.square(a-b)))/len(a)


#Just calculate the energy of one structure, maybe plot RMS force error vs system size and compare to
# Figure 1 here: https://www.researchgate.net/publication/225092546_A_Smooth_Particle_Mesh_Ewald_Method
if __name__ == "__main__":

    r_cut_real = 4.0 #idk this is random
    r_cut_neighbor = r_cut_real + 1.0 # also not sure what this should be
    error_tol = 1e-5 #on GPU OpenMM warns this is lower limit and error can start going up (should check when we do GPU)
    spline_interp_order = 5

    dump_path = r"c"


    lattice_param = 5.43 #Angstroms
    real_lat_vecs = np.array([[lattice_param,0,0],[0,0,lattice_param],[0,0,lattice_param]]) #often denoted a
    recip_lat_vecs = reciprocal_vecs(real_lat_vecs)

    #positions are (Nx3) masses, charges (Nx1), boxsize (3x1)
    positions, masses, charges, box_size = load_system(dump_path)

    #TBH no clue how to use this
    i_inds, j_inds, _ = cl.neighborlist(positions, r_cut_neighbor, unitcell=np.array([1, 1, 1]))

    U_SPME, F_SPME = PME(positions, charges, error_tol, r_cut_real, spline_interp_order)

    ground_truth_energy, ground_truth_force = brute_force_energy(positions, charges)

    rms_eng_error = rms_error(U_SPME, ground_truth_energy)
    rms_force_error = rms_error(F_SPME, ground_truth_force) #force format might not work since its Nx3


    