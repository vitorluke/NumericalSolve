import numpy as np
import scipy as sp

class MembranaElastica:
    def __init__(self, Nx:int, Ny:int, Lx:float, Ly:float):
        self.Nx = Nx
        self.Ny = Ny
        self.nunk = Nx*Ny

        self.Lx = Lx
        self.Ly = Ly

        self.K = None
        self.M = None

        self.sigma = 200
        self.e = 0.1e-3

        self.hx = Lx / (Nx - 1)
        self.hy = Ly / (Ny - 1)

        self.h = self.hx * self.hy      # Autenticar essa expressão

        self.rho = 900

    def ij2n(self, i:int, j:int):
        return i + j * self.Nx

    def build_eigen(self, mask:list[(int, int)]):
        Nx = self.Nx
        Ny = self.Ny
        nunk = self.nunk

        sigma = self.sigma
        h = self.h

        d1 = 4.0*np.ones(nunk)

        d2 = -np.ones(nunk-1)
        d3 = -np.ones(nunk-Nx)
        
        self.K = (sigma/h**2) * sp.sparse.diags([d3, d2, d1, d2, d3], [-Nx, -1, 0, 1, Nx], format='csr')
        
        # Force the eigenvalues associated to restricted points
        # to be a big number as compared to fundamental modes
        
        big_number = 10000
        Iden = big_number * sp.sparse.identity(nunk, format='csr')
        
        # Lados verticais
        
        for k in range(0,Ny):
            Ic = self.ij2n(0,k) # Left
            self.K[Ic,:], self.K[:,Ic] = Iden[Ic,:], Iden[:,Ic]
            
            Ic = self.ij2n(Nx-1,k) # Right
            self.K[Ic,:], self.K[:,Ic] = Iden[Ic,:], Iden[:,Ic]
        
        # Lados horizontais
        
        for k in range(0,Nx):
            Ic = self.ij2n(k,0) # Bottom
            self.K[Ic,:], self.K[:,Ic] = Iden[Ic,:], Iden[:,Ic]
            
            Ic = self.ij2n(k,Ny-1) # Top
            self.K[Ic,:], self.K[:,Ic] = Iden[Ic,:], Iden[:,Ic]
        
        # Processar os pontos da mascara para uma membrana circular
        for (i, j) in mask:
            Ic = self.ij2n(i,j)
            self.K[Ic,:], self.K[:,Ic] = Iden[Ic,:], Iden[:,Ic]

        # Mass matrix: Simple case, multiple of identity
        rho = self.rho
        e = self.e

        self.M = rho * e * sp.sparse.identity(nunk, format='csr')

        return self.K, self.M

    