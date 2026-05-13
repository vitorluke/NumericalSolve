import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class MembranaElastica:
    def __init__(self, Nx:int, Ny:int, R:float):
        self.Nx = Nx
        self.Ny = Ny
        self.nunk = Nx*Ny

        self.Lx = 2 * R
        self.Ly = 2 * R

        self.R = R
        
        self.hx = self.Lx / (Nx - 1)
        self.hy = self.Ly / (Ny - 1)

        self.ds = self.hx * self.hy
        
        self.e = 0.1e-3
        self.sigma = 200
        self.rho = 900

        self.K = None
        self.M = None

        self.build_laplacian()

    def ij2n(self, i:int, j:int):
        return i + j * self.Nx

    def build_laplacian(self):
        Nx = self.Nx
        Ny = self.Ny
        nunk = self.nunk

        sigma = self.sigma
        ds = self.ds

        d1 = 4.0*np.ones(nunk)
        d2 = -np.ones(nunk-1)
        d3 = -np.ones(nunk-Nx)
        
        K = sigma / ds * sp.sparse.diags([d3, d2, d1, d2, d3], [-Nx, -1, 0, 1, Nx], format='csr')
        
        big_number = 1e4
        Iden = big_number * sp.sparse.identity(nunk, format='csr')
        
        for k in range(0,Ny):
            Ic = self.ij2n(0,k)
            K[Ic,:], K[:,Ic] = Iden[Ic,:], Iden[:,Ic]
            
            Ic = self.ij2n(Nx-1,k)
            K[Ic,:], K[:,Ic] = Iden[Ic,:], Iden[:,Ic]
        
        for k in range(0,Nx):
            Ic = self.ij2n(k,0)
            K[Ic,:], K[:,Ic] = Iden[Ic,:], Iden[:,Ic]
            
            Ic = self.ij2n(k,Ny-1)
            K[Ic,:], K[:,Ic] = Iden[Ic,:], Iden[:,Ic]
        
        hx = self.hx
        hy = self.hy

        R = self.R

        for i in range(0, Nx):
            for j in range(0, Ny):
                x = i * hx - R
                y = j * hy - R

                dist_squared = x*x + y*y

                if dist_squared > R*R:
                    Ic = self.ij2n(i,j)
                    K[Ic,:], K[:,Ic] = Iden[Ic,:], Iden[:,Ic]

        rho = self.rho
        e = self.e

        M = rho * e * sp.sparse.identity(nunk, format='csr')

        self.K = K
        self.M = M

        return self.K, self.M
    
    def solve_modes(self, nmodes=10):
        assert(not(self.K is None or self.M is None))

        Lam, _ = sp.sparse.linalg.eigsh(self.K, k=nmodes*4, M=self.M, which='SM')

        Lam = np.real(Lam)
        Lam = np.sort(Lam)

        freq = np.sqrt(Lam)/(2*np.pi)

        f_01_theorical = 2.4048 / (2 * np.pi * self.R) * np.sqrt(self.sigma / (self.rho * self.e))

        freq = freq[freq > f_01_theorical / 2][:nmodes]

        return freq

    def plot_modes(self, nmodes=10):
        freq = self.solve_modes(nmodes)
        mode = np.linspace(1, nmodes, nmodes)

        plt.figure(figsize=(6,4))
        ax = plt.gca()
        ax.yaxis.get_major_formatter().set_useOffset(False)

        plt.plot(mode, freq, marker='o')

        plt.title("Modos fundamentais da membrana")

        plt.xlabel("Modo")
        plt.ylabel("Frequência (Hz)")

        plt.grid(True)

        plt.show()

def ex_02():
    R = 0.4e-2
    grids = list(map(lambda n: (20*n+1,20*n+1), range(1, 6)))

    print("-"*123)
    print("| Modo".ljust(12), end="|")

    for i in range(1, 11):
        print(f" {i}".ljust(10), end="|")

    print()
    print("-"*123)

    for (Nx, Ny) in grids:
        membrana = MembranaElastica(Nx, Ny, R)
        modes = membrana.solve_modes(10)

        print(f"| ({Nx}, {Ny})".ljust(12), end="|")

        for f in modes:
            print(f" {f:.2f}".ljust(10), end="|")

        print()

    print("-"*123)

    membrana = MembranaElastica(101, 101, R)
    membrana.plot_modes()

def main():
    ex_02()

if __name__ == "__main__":
    main()