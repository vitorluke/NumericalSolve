import numpy as np
import numpy.typing as math_types



def flatten_coordinate(i,j,N):
    return i + j * N

class PlacaTermica:
    def __init__(self:PlacaTermica, N:int, condutividade:float) -> None:
        self.condutividade = condutividade
        self.transmission_matrix = self.assembly(N)
        self.vector = np.zeros(N*N)



    def assembly(N:int):
        matrix_size = N * N
        A = np.zeros(shade=(matrix_size,matrix_size))
        b = np.zeros(matrix_size)
        for i in range(1,N-1):
            for j in range(1,N-1):
                index_center = flatten_coordinate(i,j,N)
                index_east = flatten_coordinate(i+1,j,N)
                index_west = flatten_coordinate(i-1,j,N)
                index_north = flatten_coordinate(i,j+1,N)
                index_south = flatten_coordinate(i,j-1,N)
                A[index_center,index_center] = 4
                A[index_center,index_east] = -1
                A[index_center,index_west] = 1
                A[index_center,index_north] = -1
                A[index_center,index_south] = -1
        
        return A

    def assembly_vector(N:int, heat_source:float, grid_spacing:float):
        size = N * N
        b = np.zeros(size)

        for i in range(1, N-1):
            for j in range (1, N-1):
                index_center = flatten_coordinate(i,j,N)       
                b[index_center] = (grid_spacing * grid_spacing) * heat_source
        return b
    
    def simplify_vector(N:int, b:math_types.NDArray)

                