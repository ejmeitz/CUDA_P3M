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


    return 0.5*U_direct, F_direct

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
    m_K = m/K
    num = np.exp(2*np.pi*1j*(n-1)*m_K)
    denom = np.sum([M(k+1, n)*np.exp(2*np.pi*1j*k*m_K) for k in range(n-1)])
    return  num/denom 


def calc_C(alpha, V, ms, recip_lat):
    m_star = ms[0]*recip_lat[0,:] + ms[1]*recip_lat[1,:] + ms[2]*recip_lat[2,:]
    m_sq = np.dot(m_star,m_star)
    return (1/(np.pi*V))*(np.exp(-(np.pi**2)*m_sq/(alpha**2))/m_sq)
    
def abs2(x : np.complex128):
    return (x.real**2) + (x.imag**2)

def calc_BC(alpha, V, K1, K2, K3, n, recip_lat):
    BC = np.zeros((K1,K2,K3), dtype = np.complex128)
    hs = [0.0, 0.0, 0.0]

    for m1 in range(K1):
        hs[0] = m1 if (m1 <= (K1/2)) else m1 - K1
        B1 = abs2(b(m1,K1,n))
        for m2 in range(K2):
            hs[1] = m2 if (m2 <= (K2/2)) else m2 - K2
            B2 = B1*abs2(b(m2,K2,n))
            for m3 in range(K3):
                hs[2] = m3 if (m3 <= (K3/2)) else m3 - K3
                if m1 == 0 and m2 == 0 and m3 == 0:
                    continue
                
                B3 = B2*abs2(b(m3,K3,n))
                # B = (np.abs(b(m1,K1,n)) )* (np.abs(b(m2,K2,n))) * (np.abs(b(m3,K3,n)))
                # B = np.linalg.norm(b(m1,K1,n)*b(m2,K2,n)*b(m3,K3,n))
                C = calc_C(alpha, V, hs, recip_lat)

                # m_star = hs[0]*recip_lat[0,:] + hs[1]*recip_lat[1,:] + hs[2]*recip_lat[2,:]
                # m_sq = np.dot(m_star,m_star)

                BC[m1,m2,m3] = B3*C

    return BC


def build_Q(u, n, charges, K1, K2, K3):
    N_atoms = len(charges)
    Q = np.zeros((K1, K2, K3))
    dQdr = np.zeros((N_atoms, 3, K1, K2, K3)) #deriv in real space

    for i in range(N_atoms):
        for c0 in range(n+1):
            l0 = np.round(u[i,0]) - c0 # Grid point to interpolate onto

            q_n_0 = charges[i]*M(u[i,0] - l0, n) #if 0 <= u_i0 - l0 <= n will be non-zero

            l0 += int(np.ceil(n/2)) # Shift
            if l0 < 0: # Apply PBC
                l0 += K1
            elif l0 >= K1:
                l0 -= K1

            for c1 in range(n+1):
                l1 = np.round(u[i,1]) - c1 # Grid point to interpolate onto

                q_n_1 = q_n_0*M(u[i,1] - l1, n) #if 0 <= u_i1 - l1 <= n will be non-zero


                l1 += int(np.ceil(n/2)) # Shift
                if l1 < 0: # Apply PBC
                    l1 += K2
                elif l1 >= K2:
                    l1 -= K2
                
                for c2 in range(n+1):
                    l2 = np.round(u[i,2]) - c2 # Grid point to interpolate onto

                    q_n_2 = q_n_1*M(u[i,2] - l2, n) #if 0 <= u_i1 - l1 <= n will be non-zero


                    l2 += int(np.ceil(n/2)) # Shift
                    if l2 < 0: # Apply PBC
                        l2 += K2
                    elif l2 >= K2:
                        l2 -= K2

                    Q[int(l0) ,int(l1), int(l2)] += q_n_2
                    
                    # dQdr[i, 0, int(l0), int(l1), int(l2)] += 0.0 #TODO
                    # dQdr[i, 1, int(l0), int(l1), int(l2)] += 0.0
                    # dQdr[i, 2, int(l0), int(l1), int(l2)] += 0.0
    return Q, dQdr

### TEST
def initialize_table(K1,K2,K3, recip_lat, n, alpha, V):
  
  BC = np.zeros((K1,K2,K3))
  
  for k0 in range(K1):
    h0 = k0 if k0 < K1 / 2 else k0 - K1;

    for k1 in range(K2):
      h1 = k1 if k1 < K2 / 2 else k1 - K2;

      for k2 in range(K3):

        if k0 == 0 and k1 == 0 and k2 == 0:
          continue;

        h2 = k2 if k2 < K3 / 2 else k2 - K3

        hs = [h0, h1, h2 ]
        m = np.matmul(recip_lat.T, hs)

        m_sq = np.dot(m, m)

        # B = (np.abs(b(k0,K1,n))**2 )* (np.abs(b(k1,K2,n))**2) * (np.abs(b(k2,K3,n)**2))
        B =  np.linalg.norm(b(k0, K1, n)* b(k1, K2, n) * b(k2, K3, n))

        BC[k0,k1,k2] = psi(m_sq, alpha, V) * B
  return BC

def psi(h2, alpha, V):
  b2 = np.pi * h2 / (alpha*alpha)
  b = np.sqrt(b2)
  b3 = b2 * b

  h = np.sqrt(h2)
  h3 = h2 * h

  return pow(np.pi, 9.0 / 2.0) / (3.0 * V) * h3 \
      * (np.sqrt(np.pi) * ss.erfc(b) + (1.0 / (2.0 * b3) - 1.0 / b) * np.exp(-b2))

######

#Calculate E_reciprocal for SPME
def particle_mesh(r, q, real_lat, alpha, spline_interp_order, mesh_dims):

    N_atoms = len(q)

    recip_lat = reciprocal_vecs(real_lat)
    
    K1, K2, K3 = mesh_dims
    V = vol(real_lat)
    print("ALpha: ", alpha)

    #convert coordinates to fractional
    #make this non-allocating in Julia
    #* Assumes rows of recip lat are each vector
    u = np.array([mesh_dims * np.matmul(recip_lat, r[i,:]) for i in range(r.shape[0])])

    #Fill Q array (interpolate charge onto grid)
    Q, dQdr = build_Q(u, spline_interp_order, charges, K1, K2, K3)

    print("\tQ Calculated")

    BC = calc_BC(alpha, V, K1, K2, K3, spline_interp_order, recip_lat)
    # print(np.amax(BC))
    # BC = initialize_table(K1, K2, K3, recip_lat, spline_interp_order, alpha, V)
    # print(np.amax(BC))
    #Invert Q 
    # Q_recip = np.fft.fftn(np.complex128(Q))

    # Q_recip *= BC
    
    # Q_conv_theta = np.fft.ifftn(Q_recip) #can do in place


    Q_inv = np.fft.ifftn(np.complex128(Q))

    Q_inv *= BC

    Q_conv_theta = np.fft.fftn(Q_inv)

    print(np.amax(np.abs(Q_conv_theta)))
    E_out = 0.5*np.sum(np.real(Q_conv_theta) * np.real(Q))
    # print(f"E recip: {E_out*K1*K2*K3*322}")

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

    
    pp_energy, pp_force = particle_particle(r, q, alpha, r_cut_real, real_lat_vec)
    print(f"\tP-P Energy Calculated")
    print(f"\tN-Mesh {n_mesh}")
    pm_energy, pm_force = particle_mesh(r, q, real_lat_vec, alpha, spline_interp_order, n_mesh)
    print(f"\tP-M Energy Calculated")
    self_eng = self_energy(q, alpha)

    # e_charge = 1.60217663e-19 #C
    # k = (1/(4*np.pi*scipy.constants.epsilon_0)) # N-m^2 * C^2
    # A = e_charge*e_charge*k*1e7 # kJ*Ang
    # A *= (23.06/(1.60818e-22)) # kcal/mol/Ang #fuck this unit system

    A = 332.0637128

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
    r_cut_real = 10.0 #kinda picked randomly, does this need to be less than L/2?
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


    

