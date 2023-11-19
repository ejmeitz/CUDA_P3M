import CellListMap as cl #neighbor list builder from Julia: https://m3g.github.io/CellListMap.jl/stable/python/
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
import math


def load_system(dump_path : str):
    #Just parse lammps dump file that has x y z charge mass info
    pass


#Calculate E_direct
def particle_particle(r, q, alpha, r_cut):
    N_atoms = len(q)
    U_direct = np.zeros(N_atoms)
    F_direct = np.zeros(N_atoms)

    n = 2 # WTF should this be?? I think loops should be over (0,1), feel like maybe needs to be 2 if domain is triclinic and super tilted
    for n1 in range(n): # Why cant we just use nearest mirror conventions?
        for n2 in range(n):
            for n3 in range(n):

                if n1 == 0 and n2 == 0 and n3 == 0:
                    continue 

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
    return (1/math.factorial(n-1)) * np.sum([((-1)**k)*math.comb(n,k)*np.pow(max(u-k, 0), n-1) for k in range(n+1)])

def dMdu(u,n):
    return M(u, n-1) - M(u-1, n-1)

#Calculate E_reciprocal
def particle_mesh(r, q, alpha, n_mesh, spline_interp_order):

    K1 = n_mesh; K2 = n_mesh; K3 = n_mesh

    N_atoms = len(q)


    #Fill Q array
    Q = np.zeros((K1, K2, K3))

    for k1 in range(K1):
        for k2 in range(K2):
            for k3 in range(K3):

                for i in range(N_atoms):
                    n = 2 #again not sure what n should be
                    for n1 in range(n): # Why cant we just use nearest mirror conventions?
                        for n2 in range(n):
                            for n3 in range(n):
                                Q[k1,k2,k3] += q[i] * M() * M() * M()

def self_energy(q, alpha):
   return -(alpha/np.sqrt(np.pi)) * np.sum(np.square(q))



def PME(r, q, error_tol, r_cut, spline_interp_order):

    # OpenMM parameterizes like this:
    # http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#coulomb-interaction-with-particle-mesh-ewald
    alpha = np.sqrt(-np.log(2*error_tol))/r_cut
    #Look at FFT implementation it can be better if this is a factor of some prime number
    n_mesh = np.ceil(2*alpha*error_tol/(3*np.power(error_tol, 0.2))) #assumes cubic box

    pp_energy, pp_force = particle_particle(r, q, alpha)
    pm_energy, pm_force = particle_mesh(r, q, alpha, n_mesh, spline_interp_order)
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

    r_cut = 4.0 #idk this is random
    r_cut_neighbor = r_cut + 1.0 # also not sure what this should be
    error_tol = 1e-5 #on GPU OpenMM warns this is lower limit and error can start going up (should check when we do GPU)
    spline_interp_order = 5 # paper uses 3,5,7 in tests
    dump_path = r"c"

    #positions are (Nx3) masses, charges (Nx1), boxsize (3x1)
    positions, masses, charges, box_size = load_system(dump_path)

    #TBH no clue how to use this
    i_inds, j_inds, _ = cl.neighborlist(positions, r_cut_neighbor, unitcell=np.array([1, 1, 1]))

    U_SPME, F_SPME = PME(positions, charges, error_tol, r_cut, spline_interp_order)

    ground_truth_energy, ground_truth_force = brute_force_energy(positions, charges)

    rms_eng_error = rms_error(U_SPME, ground_truth_energy)
    rms_force_error = rms_error(F_SPME, ground_truth_force) #force format might not work since its Nx3


    