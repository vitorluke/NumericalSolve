import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class MembranaElastica:
    def __init__(self, Nx:int, Ny:int, R:float):
        self.Nx = Nx
        self.Ny = Ny
        self.nunk = Nx*Ny

        self.R = R
        
        self.hx = 2 * R / (Nx - 1)
        self.hy = 2 * R / (Ny - 1)

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
    
    def solve_forced_coefficients(self, omega_star, beta, nmodes=10):
        Lam, Phi = sp.sparse.linalg.eigsh(self.K, k=nmodes*4, M=self.M, which='SM')
        
        freqs_all = np.sqrt(np.real(Lam)) / (2 * np.pi)
        f01_theory = 2.4048 / (2 * np.pi * self.R) * np.sqrt(self.sigma / (self.rho * self.e))
        
        mask = freqs_all > f01_theory / 2
        Phi = Phi[:, mask][:, :nmodes]
        omega_n = 2 * np.pi * freqs_all[mask][:nmodes]

        Z = np.zeros(self.nunk)
        for j in range(self.Ny):
            for i in range(self.Nx):
                xi = i / (self.Nx - 1)
                yi = j / (self.Ny - 1)
                Z[self.ij2n(i, j)] = (xi - 0.5)**2 + (yi - 0.5)**2

        alphas = Phi.T @ Z

        denominador = np.sqrt((-omega_star**2 + omega_n**2)**2 + (beta * omega_star)**2)
        ci = alphas / denominador

        return ci, alphas
    
    def solve_modes(self, nmodes=10):
        assert(not(self.K is None or self.M is None))

        k = min(nmodes*4, self.Nx*self.Ny - 1)

        Lam, _ = sp.sparse.linalg.eigsh(self.K, k=k, M=self.M, which='SM')

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

    print("-"*124)
    print("| Modo".ljust(13), end="|")

    for i in range(1, 11):
        print(f" {i}".ljust(10), end="|")

    print()
    print("-"*124)

    for (Nx, Ny) in grids:
        membrana = MembranaElastica(Nx, Ny, R)
        modes = membrana.solve_modes(10)

        print(f"| ({Nx}, {Ny})".ljust(13), end="|")

        for f in modes:
            print(f" {f:.2f}".ljust(10), end="|")

        print()

    print("-"*124)

    membrana = MembranaElastica(101, 101, R)
    membrana.plot_modes()

def ex_04():
    R = 0.4e-2
    Nx, Ny = 51, 51
    membrana = MembranaElastica(Nx, Ny, R)
    
    beta = 0.1
    omega_star = 2 * np.pi * 5000
    
    ci, alphas = membrana.solve_forced_coefficients(omega_star, beta)
    
    print("Representação do termo forçante (alphas) e Coeficientes de Resposta (ci):")
    print("-" * 60)
 
    print(f"{'Modo':<10} | {'alpha_i (Força)':<20} | {'c_i (Resposta)':<20}")
    print("-" * 60)
 
    for i in range(len(ci)):
        print(f"{i+1:<10} | {alphas[i]:<20.6e} | {ci[i]:<20.6e}")
    
def ex_05():
    R = 0.4e-2
    Nx, Ny = 51, 51
    membrana = MembranaElastica(Nx, Ny, R)
    
    omega_star_hat_range = np.logspace(np.log10(0.5), np.log10(100), 1000)
    betas = [0.01, 0.1, 1]
    nmodes = Nx*Ny-1

    freqs = membrana.solve_modes(nmodes)
    omega_i_fisico = 2 * np.pi * freqs

    omega_ref = (2.4048 / membrana.R) * np.sqrt(membrana.sigma / (membrana.rho * membrana.e))

    omega_i_hat = omega_i_fisico / omega_ref

    Lam, Phi = sp.sparse.linalg.eigsh(membrana.K, k=nmodes, M=membrana.M, which='SM')
    f01_theory = 2.4048 / (2 * np.pi * membrana.R) * np.sqrt(membrana.sigma / (membrana.rho * membrana.e))
    mask = (np.sqrt(np.real(Lam))/(2*np.pi)) > f01_theory / 2
    Phi_physical = Phi[:, mask][:, :nmodes]

    Z = np.zeros(membrana.nunk)
    for j in range(membrana.Ny):
        for i in range(membrana.Nx):
            xi, yi = i / (membrana.Nx - 1), j / (membrana.Ny - 1)
            Z[membrana.ij2n(i, j)] = (xi - 0.5)**2 + (yi - 0.5)**2

    alphas = Phi_physical.T @ Z

    plt.figure(figsize=(10, 6))
    colors = ['red', 'magenta', 'green']

    for beta, color in zip(betas, colors):
        Ae_list = []
        for ws_hat in omega_star_hat_range:
            denominador = np.sqrt((-ws_hat**2 + omega_i_hat**2)**2 + (beta * ws_hat)**2)
            ci = alphas / denominador
            
            Ae = 0.25 * np.sum((ci**2) * (omega_i_hat**2))
            Ae_list.append(Ae)
        
        plt.loglog(omega_star_hat_range, Ae_list, label=f'beta = {beta}', color=color)

    plt.title("Energia Elástica Média vs Frequência Forçante Adimensional")
    plt.xlabel("$\hat{\omega}_*$")
    plt.ylabel("Mean Energy ($A_e$)")
    plt.ylim(1e-1, 2e3)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.show()

def main():
    ex_02()

if __name__ == "__main__":
    main()