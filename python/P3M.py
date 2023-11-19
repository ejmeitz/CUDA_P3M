import CellListMap as cl #neighbor list builder from Julia: https://m3g.github.io/CellListMap.jl/stable/python/
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss


def load_system(dump_path : str):
    #Just parse lammps dump file that has x y z charge mass info
    pass


#Calculate E_direct
def particle_particle(r, q, alpha, r_cut):
    N_atoms = len(q)
    U_direct = np.zeros(N_atoms)
    F_direct = np.zeros(N_atoms)

    n = 2 # WTF should this be?? I think loops should be over (0,1)
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

#Calculate E_reciprocal
def particle_mesh(r, charges, alpha, n_mesh):
    pass

def self_energy(q, alpha):
   return -(alpha/np.sqrt(np.pi)) * np.sum(np.square(q))


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
    dump_path = r"c"

    # OpenMM parameterizes like this:
    # http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#coulomb-interaction-with-particle-mesh-ewald
    alpha = np.sqrt(-np.log(2*error_tol))/r_cut
    n_mesh = 2*alpha*error_tol/(3*np.power(error_tol, 0.2)) #assumes cubic box

    #positions are (Nx3) masses, charges (Nx1), boxsize (3x1)
    positions, masses, charges, box_size = load_system(dump_path)

    #TBH no clue how to use this
    i_inds, j_inds, _ = cl.neighborlist(positions, r_cut_neighbor, unitcell=np.array([1, 1, 1]))

    pp_energy, pp_force = particle_particle(positions, charges, alpha)
    pm_energy, pm_force = particle_mesh(positions, charges, alpha, n_mesh)
    self_eng = self_energy(positions, charges)
    U_SPME = pp_energy + pm_energy + self_eng
    F_SPME = pp_force + pm_force

    ground_truth_energy, ground_truth_force = brute_force_energy(positions, charges)

    rms_eng_error = rms_error(U_SPME, ground_truth_energy)
    rms_force_error = rms_error(F_SPME, ground_truth_force) #force format might not work since its Nx3


    