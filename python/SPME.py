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

                            # term1 = ((2/np.sqrt(np.pi)) *dist_ijn*alpha*np.exp((-alpha*alpha*dist_ijn*dist_ijn)) + ss.erfc(alpha*dist_ijn))/(dist_ijn**3)
                            # F_ij = q[i] * q[j] * term1 * r_ijn

                            #Equivalent to ^
                            F_ij = q[i] * q[j] * ((ss.erfc(alpha*dist_ijn)/(dist_ijn**2)) + (2*alpha*np.exp((-alpha*alpha*dist_ijn*dist_ijn))/(np.sqrt(np.pi)*dist_ijn)))
                            r_hat = r_ijn / dist_ijn 
                            F_ij = F_ij*r_hat

                            F_direct[i,:] += F_ij
                            F_direct[j,:] -= F_ij


    return 0.5*U_direct, 0.5*F_direct

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
    if n % 2 == 1:
        if 2 * np.abs(m) == K:
            return 0
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
                C = calc_C(alpha, V, hs, recip_lat)
                BC[m1,m2,m3] = B3*C

    return BC


def build_Q(u, n, charges, K1, K2, K3, recip_lat):
    N_atoms = len(charges)
    Q = np.zeros((K1, K2, K3))
    dQdr = np.zeros((N_atoms, 3, K1, K2, K3)) #deriv in real space

    for i in range(N_atoms):
        for c0 in range(n+1):
            l0 = np.round(u[i,0]) - c0 # Grid point to interpolate onto

            M0 = M(u[i,0] - l0, n)
            q_n_0 = charges[i]*M0 #if 0 <= u_i0 - l0 <= n will be non-zero
            dM0 = dMdu(u[i,0] - l0,n)

            l0 += int(np.ceil(n/2)) # Shift
            if l0 < 0: # Apply PBC
                l0 += K1
            elif l0 >= K1:
                l0 -= K1

            for c1 in range(n+1):
                l1 = np.round(u[i,1]) - c1 # Grid point to interpolate onto

                M1 = M(u[i,1] - l1, n)
                q_n_1 = q_n_0*M1 #if 0 <= u_i1 - l1 <= n will be non-zero
                dM1 = dMdu(u[i,1] - l1,n)


                l1 += int(np.ceil(n/2)) # Shift
                if l1 < 0: # Apply PBC
                    l1 += K2
                elif l1 >= K2:
                    l1 -= K2
                
                for c2 in range(n+1):
                    l2 = np.round(u[i,2]) - c2 # Grid point to interpolate onto

                    M2 = M(u[i,2] - l2, n)
                    q_n_2 = q_n_1*M2 #if 0 <= u_i1 - l1 <= n will be non-zero
                    dM2 = dMdu(u[i,2] - l2,n)

                    l2 += int(np.ceil(n/2)) # Shift
                    if l2 < 0: # Apply PBC
                        l2 += K2
                    elif l2 >= K2:
                        l2 -= K2

                    Q[int(l0) ,int(l1), int(l2)] += q_n_2
                    
                    #*Does it matter that l0,l1,l2 is also a function of r_ia
                    #*This looks like its probably equivalent to some matrix multiply
                    dQdr[i, 0, int(l0), int(l1), int(l2)] = charges[i]*(K1*recip_lat[0,0]*dM0*M1*M2 + K2*recip_lat[1,0]*dM1*M0*M2 + K3*recip_lat[2,0]*dM2*M0*M1)
                    dQdr[i, 1, int(l0), int(l1), int(l2)] = charges[i]*(K1*recip_lat[0,1]*dM0*M1*M2 + K2*recip_lat[1,1]*dM1*M0*M2 + K3*recip_lat[2,1]*dM2*M0*M1)
                    dQdr[i, 2, int(l0), int(l1), int(l2)] = charges[i]*(K1*recip_lat[0,2]*dM0*M1*M2 + K2*recip_lat[1,2]*dM1*M0*M2 + K3*recip_lat[2,2]*dM2*M0*M1)
    return Q, dQdr


#Calculate E_reciprocal for SPME
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
    Q, dQdr = build_Q(u, spline_interp_order, charges, K1, K2, K3, recip_lat)

    BC = calc_BC(alpha, V, K1, K2, K3, spline_interp_order, recip_lat)

    print(np.max(np.abs(Q)))
    # print(np.max(np.abs(BC)))


    Q_inv = np.fft.ifftn(np.complex128(Q))
    Q_inv *= BC
    Q_conv_theta = np.fft.fftn(Q_inv)

    # Q_inv = np.fft.fftn(np.complex128(Q))
    # Q_inv *= BC
    # Q_conv_theta = np.fft.ifftn(Q_inv)


    E_out = 0.5*np.sum(np.real(Q_conv_theta) * np.real(Q))

    F_out = np.zeros((N_atoms, 3))
    for i in range(N_atoms):
        for dir in range(3):
            F_out[i, dir] = -np.sum(dQdr[i,dir,:,:,:] * Q_conv_theta) #do this without allocating
 

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
    pm_energy, pm_force = particle_mesh(r, q, real_lat_vec, alpha, spline_interp_order, n_mesh)
    self_eng = self_energy(q, alpha)

    # e_charge = scipy.constants.elementary_charge #C
    # k = (1/(4*np.pi*scipy.constants.epsilon_0)) # N-m^2 * C^2
    # A = e_charge*e_charge*k*1e10 # J*Ang
    # eV_per_J = 1.602176634e-19 #not sure on the accuracy of this
    # kcal_mol_per_eV = 23.06054194 #not sure on the accuracy of this
    # kcal_per_joule = 1/4184
    # Na = 6.02214076e23
    # A *= kcal_per_joule*Na # kcal/mol/Ang #fuck this unit system

    A = 332.0637132991921

    print(f"PP Energy: {np.sum(pp_energy*A)}")
    print(f"PM Energy {pm_energy*A}")
    print(f"Self Energy {self_eng*A}")
    print(f"Total Ewald Eng {(np.sum(pp_energy) + pm_energy + self_eng)*A}")

    U_SPME_total = (np.sum(pp_energy) + pm_energy + self_eng)*A
    F_SPME = (pp_force + pm_force)*A
    # This artifact can be avoided by removing the average net force
    # from each atom at each step of the simulation
    # print(A*pp_force[0,:])
    # print(A*pm_force[0,:])
    #TODO
    avg_net_force = 0.0 #tf u take the average of?


    return U_SPME_total, F_SPME - avg_net_force


def rms_error(a, b):
    return np.sqrt(np.sum(np.square(a-b)))/len(a)


#Just calculate the energy of one structure, maybe plot RMS force error vs system size and compare to
# Figure 1 here: https://www.researchgate.net/publication/225092546_A_Smooth_Particle_Mesh_Ewald_Method
if __name__ == "__main__":

    r_cut_lj = 7.0 #needs to be less then 8 for box size w/ 3 UC
    r_cut_real = 7.0 #kinda picked randomly, does this need to be less than L/2?
    r_cut_neighbor = r_cut_real + 1.0 #not sure what this should be
    error_tol = 1e-4 #GPU OpenMM warns 5e-5 is lower limit and error can start going up (should check when we do GPU)
    spline_interp_order = 6 #OpenMM uses 5


    dump_path = os.path.join(r"../test_data/salt_sim/dump.atom")


    lattice_param = 5.62 #Angstroms
    N_uc = 3
    real_lat_vecs = np.array([[N_uc*lattice_param,0,0],[0,N_uc*lattice_param,0],[0,0,N_uc*lattice_param]]) #often denoted a
    recip_lat_vecs = reciprocal_vecs(real_lat_vecs)

    #positions are (Nx3) masses, charges (Nx1), boxsize (3x1)
    N_steps = 11
    positions, forces_lmp, eng_lmp, masses, charges, box_sizes = load_system(dump_path, N_steps)
    for i in range(N_steps):
        print(f"ON LOOP ITERATION {i}")
        U_LJ, F_LJ = lj_energy_loop(positions[:,:,i], charges, box_sizes, r_cut_lj)

        #TBH no clue how to use this
        # i_inds, j_inds, _ = cl.neighborlist(positions[:,:,i], r_cut_neighbor, unitcell=np.array([1, 1, 1]))
        U_SPME_total, F_SPME = PME(positions[:,:,i], charges, real_lat_vecs, error_tol, r_cut_real, spline_interp_order)
        # print(f"\t PME Energy Calculated")
        print(f"U_total_PME {U_SPME_total}")
        print(f"U_total {U_SPME_total + np.sum(U_LJ)}")
        print(f"Force on Atom 0: {F_SPME[0,:] + F_LJ[0,:]}")


    

