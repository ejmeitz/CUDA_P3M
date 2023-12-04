import numpy as np


# 2D direct summation 
def directSum_2D(r, q, a, n=10):
    U = 0 
    charges = len(q)
    vecs = np.arange(-n, n+1)
    for idx in range(charges):
        for idx2 in range(charges):
            for i in vecs:
                for j in vecs:
                    if i == 0 and j == 0 and idx == idx2:
                        continue
                    dist = np.linalg.norm(r[idx] - r[idx2] + i*a[0] + j*a[1])
                    if dist == 0:
                        continue 
                    U += q[idx]*q[idx2]/dist
    return U/2

def directSum_3D(r, q, a, n=2):
    U = 0 
    charges = len(q)
    vecs = np.arange(-n, n+1)
    for idx in range(charges):
        for idx2 in range(charges):
            for i in vecs:
                for j in vecs:
                    for k in vecs:
                        if i == 0 and j == 0 and k == 0 and idx == idx2:
                            continue
                        dist = np.linalg.norm(r[idx] - r[idx2] + i*a[0] + j*a[1] + k*a[2])
                        if dist == 0:
                            continue
                        U += q[idx]*q[idx2]/dist
    return U/2