import numpy as np
import numpy.typing as npt

def jacobi_iteration(A:npt.ArrayLike, b:npt.ArrayLike, x:npt.ArrayLike) -> npt.ArrayLike:
    n = len(b)
    out_x = np.zeros(n) 
    for i in range(n):
        reciprocal_aii = 1/A[i][i]
        sum = 0
        for j in range(n):
            if j != i:
                continue
            sum += A[i][j] * x[j]
        out_x[i] = reciprocal_aii * (b[i] - sum)
    return out_x

"""
Before calling, guarantee that D is a vector, not a matrix.
"""
def fast_jacobi_iteration(D:npt.ArrayLike, R:npt.ArrayLike, b:npt.ArrayLike, x:npt.ArrayLike) -> npt.ArrayLike:
    return (b-np.dot(R,x)) / D


def gauss_seidel_iteration(A: npt.ArrayLike, b: npt.ArrayLike, x: npt.ArrayLike) -> npt.ArrayLike:
    n = len(b)
    out_x = x.copy() 
    for i in range(n):
        sum_term = 0
        for j in range(n):
            if j == i:
                continue
            sum_term += A[i][j] * out_x[j]
                
        out_x[i] = (b[i] - sum_term) / A[i][i]
    return out_x


