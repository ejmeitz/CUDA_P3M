# import CellListMap as cl #neighbor list builder from Julia: https://m3g.github.io/CellListMap.jl/stable/python/
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
import scipy.signal as sig
import scipy
import os
import math

from helper import *

#similar code: https://github.com/jht0664/structurefactor_spme/blob/master/run_sq.py

#Calculate E_direct
def particle_particle(r, q, alpha, r_cut, real_lat):
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
                            U_direct[i] += q[i] * q[j] * ss.erfc(alpha*dist_ijn) / dist_ijn

                            F_ij = q[i] * q[j] #TODO  #TODO

                            r_hat = r_ijn / dist_ijn 
                            F_ij = F_ij*r_hat

                            F_direct[i,:] += F_ij
                            F_direct[j,:] -= F_ij #on GPU this is not worth doing

    return U_direct, F_direct

#From appendix B
def M(u, n):
    if n > 2:
        return (u/(n-1))*M(u,n-1) + ((n-u)/(n-1))*M(u-1,n-1)
    elif n == 2:
        if u >= 0 and u <= 2:
            return 1 - np.abs(u-1)
        else:
            return 0
    else:
        print("Shouldn't be here")

#equivalent, time to see whats faster
# def M(u, n):
#     return (1/math.factorial(n-1)) * np.sum([((-1)**k)*math.comb(n,k)*np.power(max(u-k, 0), n-1) for k in range(n+1)])

def dMdu(u,n):
    return M(u, n-1) - M(u-1, n-1)

# Make this non-allocating
# Make this not use np.exp(j) just manage cos() and sin() manually
def b(m, K, n): #n is spline order
    return np.exp(2*np.pi*1.0j*(n-1)*m/K) / sum([M(k+1, n)*np.exp(2*np.pi*1.0j*m*k/K) for k in range(n-1)])


def calc_C(alpha, V, ms, recip_lat):
    m_star = ms[0]*recip_lat[0,:] + ms[1]*recip_lat[1,:] + ms[2]*recip_lat[2,:]
    m_sq = np.dot(m_star,m_star)
    return (1/(np.pi*V))*(np.exp(-(np.pi**2)*m_sq/(alpha**2))/m_sq)
    

def calc_BC(alpha, V, K1, K2, K3, n, recip_lat, real_lat):
    BC = np.zeros((K1,K2,K3), dtype = np.complex128)
    hs = [0.0, 0.0, 0.0]
    for m1 in range(K1):
        hs[0] = m1 if (m1 < (K1/2)) else m1 - K1
        for m2 in range(K2):
            hs[1] = m2 if (m2 < (K2/2)) else m2 - K2
            for m3 in range(K3):
                hs[2] = m3 if (m3 < (K3/2)) else m3 - K3
                if m1 == 0 and m2 == 0 and m3 == 0:
                    continue
                
                B = np.abs(b(m1,K1,n) * b(m2,K2,n) * b(m3,K3,n)) #& norm == abs here?
                C = calc_C(alpha, V, hs, recip_lat)

                m_star = hs[0]*recip_lat[0,:] + hs[1]*recip_lat[1,:] + hs[2]*recip_lat[2,:]
                m_sq = np.dot(m_star,m_star)

                BC[m1,m2,m3] = B * psi(m_sq, V, alpha)

    return BC

#& prob adds error, but saves having to eval this func a fuck ton 
def calc_bsplines(n_spline_vals, n):
    b_spline_arr = np.empty(n_spline_vals)
    for i in range(n_spline_vals):
        b_spline_arr[i] = M(n/np.float_(n_spline_vals)*np.float_(i+1), n)
    return b_spline_arr

def build_Q(u, n, charges, K1, K2, K3, spline_vals):
    N_atoms = len(charges)
    Q = np.zeros((K1, K2, K3))
    dQdr = np.zeros((N_atoms, 3, K1, K2, K3)) #deriv in real space

    arg = np.empty(3) # distance between (original pt - nearpt), adding one element from range(spline_order)
    for j in range(N_atoms):
        nearpt=np.int_(np.floor(u[j,:]))
        # only need to go to k=0,n-1, for k=n, arg > n, so don't consider this
        for k1 in range(n):
            n1 = nearpt[0]-k1
            arg[0] = u[j,0]-np.float_(n1)
            n1 = np.mod(n1,K1)
            for k2 in range(n):
                n2 = nearpt[1]-k2
                arg[1] = u[j,1]-np.float_(n2)
                n2 = np.mod(n2,K2)
                for k3 in range(n):
                    n3 = nearpt[2]-k3
                    arg[2] = u[j,2]-np.float_(n3)
                    n3 = np.mod(n3,K3)
                    #orig didnt have the -1 but I needed that to avoid edge case
                    splindex = np.ceil((arg/n)*np.float_(len(spline_vals))) - 1
                    splindex = np.int_(splindex) 
                    Q[n1,n2,n3] += charges[j]*spline_vals[splindex[0]]*spline_vals[splindex[1]]*spline_vals[splindex[2]]
                    
                    dQdr[i, 0, k1, k2, k3] += 0.0 #TODO
                    dQdr[i, 1, k1, k2, k3] += 0.0
                    dQdr[i, 2, k1, k2, k3] += 0.0
    return Q, dQdr


############## TEST

#From PME implementation in C++, gotta figure out horner method for other orders
def compute_spline(u): #ONLY WORKS FOR ORDER 4
    floor_u = np.floor(u);
    x = u - floor_u + np.float_(4) - 1.0;

    #Compute splines and their derivatives using Horner's method.
    a = np.array([
        32.0 / 3.0, -8.0, 2.0, -1.0 / 6.0,
        -22.0 / 3.0, 10.0, -4.0, 1.0 / 2.0,
        2.0 / 3.0, -2.0, 2.0, -1.0 / 2.0,
        0.0, 0.0, 0.0, 1.0 / 6.0])
    
    s = np.zeros(4)
    ds = np.zeros(4)
    for k in range(4):
        s[k] = a[4*k] + (a[4*k+1] + (a[4*k+2] + a[4*k+3] * x) * x) * x;
        ds[k] = a[4*k+1] + (2.0 * a[4*k+2] + 3.0 * a[4*k+3] * x) * x;
        x = x - 1

    return s, ds, floor_u

#REQUIRES ORDER = 4
def build_Q2(u, q, order, K1, K2, K3):
    assert order == 4
    M_vals = np.zeros((3, len(q), order))
    dM_vals = np.zeros((3, len(q), order))
    Q = np.zeros((K1, K2, K3))
    dQdr = 0.0

    for i in range(len(q)):
        q_n = q[i]
        u_i = u[i]

        floor_u = np.zeros(3)
        for d in range(3): #2 is dimensions
            M_vals[d,i,:], dM_vals[d,i,:], floor_u[d] = compute_spline(u_i[d])

        for c0 in range(order):
            l0 = floor_u[0] + c0 - order + 1
            #           k0 = l0 < 0 ? l0 + K1 : l0
            k0 = l0 + K1 if l0 < 0 else l0

            q_n_s0 = q_n * M_vals[0,i,c0]

            for c1 in range(order):
                l1 = floor_u[1] + c1 - order + 1
                #             k1 = l1 < 0 ? l1 + K2 : l1
                k1 = l1 + K2 if l1 < 0 else l1
                
                q_n_s0_s1 = q_n_s0 * M_vals[1,i,c1]

                for c2 in range(order):
                    l2 = floor_u[2] + c2 - order + 1
                    k2 = l2 + K3 if l2 < 0 else l2


                    Q[int(k0), int(k1), int(k2)] += q_n_s0_s1 * M_vals[2,i,c1]

    return Q, dQdr

#what in the fuck is this??????????
def psi(h2, V, alpha):
  b2 = np.pi * h2 / (alpha**2)
  b = np.sqrt(b2)
  b3 = b2 * b

  h = np.sqrt(h2)
  h3 = h2 * h

  return pow(np.pi, 9.0 / 2.0) / (3.0 * V) * h3 \
      * (np.sqrt(np.pi) * ss.erfc(b) + (1.0 / (2.0 * b3) - 1.0 / b) * np.exp(-b2));

####################

#Calculate E_reciprocal
def particle_mesh(r, q, real_lat, alpha, spline_interp_order, mesh_dims):

    N_atoms = len(q)

    recip_lat = reciprocal_vecs(real_lat)
    
    K1, K2, K3 = mesh_dims

    V = vol(real_lat)

    #convert coordinates to fractional
    #make this non-allocating in Julia
    #* Assumes rows of recip lat are each vector
    u = np.array([mesh_dims * np.matmul(recip_lat, r[i,:]) for i in range(r.shape[0])])

    #Fill Q array (interpolate charge onto grid)
    spline_vals = calc_bsplines(100000, spline_interp_order)
    Q, dQdr = build_Q(u, spline_interp_order, charges, K1, K2, K3, spline_vals)
    #should be equiv & faster than mine but only works order 4
    # Q, dQdr = build_Q2(u, charges, spline_interp_order, K1, K2, K3)
    # plt.imshow(Q[:,:,2])

    print("\tQ Calculated")

    BC = calc_BC(alpha, V, K1, K2, K3, spline_interp_order, recip_lat, real_lat)
    # sio = spline_interp_order #just alias

    # v = [0.0,0.0,0.0] #pre-alloc
    
    # for k1 in range(K1):
    #     for k2 in range(K2):
    #         for k3 in range(K3):

                 
    #             #* not sure this loop is right
    #             #& my current interpretation is that for a given grid point (k1,k2,k3) get contribution from all other grid pts
    #             for i in range(N_atoms):
    #                 for p1 in range(K1):
    #                     for p2 in range(K2):
    #                         for p3 in range(K3):                            
    #                             u0 = u[i,0] - k1 - p1*K1; u1 = u[i,1]- k2 - p2*K2; u2 = u[i,2] - k3 - p3*K3

    #                             I = dMdu(u0,sio) * M(u1,sio) * M(u2,sio)
    #                             II = M(u0,sio) * dMdu(u1,sio) * M(u2,sio)
    #                             III = M(u0,sio) * M(u1,sio) * dMdu(u2,sio)

    #                             v[0] = K1*I; v[1] = K2*II; v[2] = K3*III

    #                             #dQdr_i0
    #                             dQdr[i, 0, k1, k2, k3] += np.sum(recip_lat[:,0] * v)
    #                             #dQdr_i1
    #                             dQdr[i, 1, k1, k2, k3] += np.sum(recip_lat[:,1] * v)
    #                             #dQdr_i2
    #                             dQdr[i, 2, k1, k2, k3] += np.sum(recip_lat[:,2] * v)

    #                             #Real space charge interpolated onto mesh
    #                             Q[k1,k2,k3] += q[i] * M(u0,sio) * M(u1,sio) * M(u2,sio)
                                

    #Invert Q (do in place on GPU??)
    Q_recip = np.fft.fftn(np.complex_(Q))
    Q_recip *= BC

    QBC_real_space = np.fft.ifftn(Q_recip)
    E_out = 0.0
    for k1 in range(K1):
        for k2 in range(K2):
            for k3 in range(K3):
                E_out += np.real(QBC_real_space[k1,k2,k3]) * np.real(Q_recip[k1,k2,k3])



    F_out = np.zeros((N_atoms, 3))
    # for i in range(N_atoms):
    #     for dir in range(3):
            # F_out[i, dir] = -np.sum(dQdr[i,dir,:,:,:] * theta_rec_conv_Q) #do this without allocating
 

    return E_out, F_out


def self_energy(q, alpha):
    return -(alpha/np.sqrt(np.pi)) * np.sum(np.square(q))


def PME(r, q, real_lat_vec, error_tol, r_cut_real, spline_interp_order):


    L = np.array([np.linalg.norm(real_lat_vec[i,:]) for i in range(real_lat_vec.shape[0])])
    # OpenMM parameterizes like this:
    # http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#coulomb-interaction-with-particle-mesh-ewald
    alpha = np.sqrt(-np.log(2*error_tol))/r_cut_real
    #Look at FFT implementation it can be better if this is a factor of some prime number
    n_mesh = np.ceil(2*alpha*L/(3*np.power(error_tol, 0.2))).astype(np.int64)

    # n_mesh = [6,6,6] #idk its slow as shit with that eqn ^

    pp_energy, pp_force = particle_particle(r, q, alpha, r_cut_real, real_lat_vec)
    print(f"\tP-P Energy Calculated")
    print(f"\tN-Mesh {n_mesh}")
    # pm_energy, pm_force = particle_mesh(r, q, real_lat_vec, alpha, spline_interp_order, n_mesh)
    print(f"\tP-M Energy Calculated")
    self_eng = self_energy(q, alpha)

    pm_energy = 0.0
    pm_force = 0.0

    # e_charge = 1.60217663e-19 #C
    # k = (1/(4*np.pi*scipy.constants.epsilon_0)) # N-m^2 * C^2
    # A = e_charge*e_charge*k*1e7 # kJ*Ang
    # A *= (23.06/(1.60818e-22)) # kcal/mol/Ang #fuck this unit system

    A = 332.218 #from file:///C:/Users/ejmei/Downloads/Lecture%2012%20-%20MD-1.pdf same as ^

    print(f"PP Energy: {np.sum(pp_energy*A)}")
    print(f"PM Energy {np.sum(pm_energy)*A}")
    print(f"Self Energy {np.sum(self_eng)*A}")
    print(f"Total Ewald Eng {(np.sum(pp_energy) + np.sum(pm_energy) + np.sum(self_eng))*A}")

    U_SPME = pp_energy + pm_energy + self_eng
    F_SPME = pp_force + pm_force

    # This artifact can be avoided by removing the average net force
    # from each atom at each step of the simulation

    #TODO
    avg_net_force = 0.0 #tf u take the average of?


    return U_SPME, F_SPME - avg_net_force


def rms_error(a, b):
    return np.sqrt(np.sum(np.square(a-b)))/len(a)


#Just calculate the energy of one structure, maybe plot RMS force error vs system size and compare to
# Figure 1 here: https://www.researchgate.net/publication/225092546_A_Smooth_Particle_Mesh_Ewald_Method
if __name__ == "__main__":

    r_cut_lj = 7.0 #needs to be less then 8 for box size w/ 3 UC
    r_cut_real = 10.0 #kinda picked randomly
    r_cut_neighbor = r_cut_real + 1.0 #not sure what this should be
    error_tol = 1e-4 #GPU OpenMM warns 5e-5 is lower limit and error can start going up (should check when we do GPU)
    spline_interp_order = 5 #OpenMM uses 5


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
        U_SPME, F_SPME = PME(positions[:,:,i], charges, real_lat_vecs, error_tol, r_cut_real, spline_interp_order)
        # print(f"\t PME Energy Calculated")


        # print(U_SPME)
        # rms_eng_error = rms_error(U_SPME + U_LJ, forces_lmp)p
        # rms_force_error = rms_error(F_SPME + F_LJ, eng_lmp) #force format might not work since its Nx3


    

