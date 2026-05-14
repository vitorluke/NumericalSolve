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
    
    def solve_modes(self, nmodes=10):
        assert(not(self.K is None or self.M is None))

        k = min(nmodes*2, self.Nx*self.Ny - 1)

        Lam, modes = sp.sparse.linalg.eigsh(self.K, k=k, M=self.M, which='SM')

        Lam = np.real(Lam)
        Lam = np.sort(Lam)

        omega = np.sqrt(Lam)
        freq = omega / (2 * np.pi)

        f01 = 2.4048 / (2 * np.pi * self.R) * np.sqrt(self.sigma / (self.rho * self.e))

        filter = freq > (f01 / 2)

        freq = freq[filter][:nmodes]
        omega = omega[filter][:nmodes]
        modes = modes[:, filter][:, :nmodes]

        return freq, omega, modes

    def plot_modes(self, nmodes=10):
        freq, _, _ = self.solve_modes(nmodes)

        k = len(freq)
        mode = np.linspace(1, k, k)

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

    print("Tabela das frequências fundamentais da membrana (Hz)")

    print("-"*124)
    print("| Modo".ljust(13), end="|")

    for i in range(1, 11):
        print(f" {i}".ljust(10), end="|")

    print()
    print("-"*124)

    for (Nx, Ny) in grids:
        membrana = MembranaElastica(Nx, Ny, R)
        freq, _, _ = membrana.solve_modes(10)

        print(f"| ({Nx}, {Ny})".ljust(13), end="|")

        for f in freq:
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
    f_star = 5000
    omega_star = 2 * np.pi * f_star
    n_solicitado = 10

    freqs, omegas, modes = membrana.solve_modes(nmodes=n_solicitado)

    modes_norm = np.zeros_like(modes)
    for i in range(modes.shape[1]):
        phi_i = modes[:, i]
        norm_m = np.sqrt(phi_i.T @ (membrana.M @ phi_i))
        modes_norm[:, i] = phi_i / norm_m

    Z = np.zeros(membrana.nunk)
    for j in range(membrana.Ny):
        for i in range(membrana.Nx):
            xi = i / (membrana.Nx - 1)
            yi = j / (membrana.Ny - 1)
            Z[membrana.ij2n(i, j)] = (xi - 0.5)**2 + (yi - 0.5)**2

    alphas = modes_norm.T @ Z

    denominador = np.sqrt((omegas**2 - omega_star**2)**2 + (beta * omega_star)**2)
    ci = alphas / denominador

    print(f"omega* = {omega_star} rad/s e beta = {beta}")
    print("-" * 80)
    print(f"| {'Modo':<6} | {'Freq Nat (Hz)':<15} | {'alpha_i (Força)':<20} | {'c_i (Resposta)':<20}")
    print("-" * 80)

    for i in range(len(freqs)):
        print(f"{i+1:<6} | {freqs[i]:<15.2f} | {alphas[i]:<20.6e} | {ci[i]:<20.6e}")
    print("-" * 80)

    return ci, alphas
    
def ex_05():
    R = 0.4e-2
    Nx, Ny = 51, 51
    membrana = MembranaElastica(Nx, Ny, R)
    
    ws_hat_range = np.logspace(np.log10(0.5), np.log10(100), 500)
    betas = [0.01, 0.1, 1]
    
    omega_ref = (2.4048 / R) * np.sqrt(membrana.sigma / (membrana.rho * membrana.e))

    freqs, omegas, modes = membrana.solve_modes(nmodes=membrana.nunk-1)

    modes_norm = np.zeros_like(modes)
    for i in range(modes.shape[1]):
        phi_i = modes[:, i]
        norm_m = np.sqrt(phi_i.T @ (membrana.M @ phi_i))
        modes_norm[:, i] = phi_i / norm_m

    Z = np.zeros(membrana.nunk)
    for j in range(membrana.Ny):
        for i in range(membrana.Nx):
            xi, yi = i / (Nx - 1), j / (Ny - 1)
            Z[membrana.ij2n(i, j)] = (xi - 0.5)**2 + (yi - 0.5)**2
    
    alphas = modes_norm.T @ Z

    omega_n_hat = omegas / omega_ref

    plt.figure(figsize=(10, 6))
    colors = ['red', 'magenta', 'green']

    for beta, color in zip(betas, colors):
        Ae_list = []
        for ws_hat in ws_hat_range:
            denominador = np.sqrt((omega_n_hat**2 - ws_hat**2)**2 + (beta * ws_hat)**2)
            ci = alphas / denominador
            
            Ae = 0.25 * np.sum((ci**2) * (omega_n_hat**2))
            Ae_list.append(Ae)
        
        plt.loglog(ws_hat_range, Ae_list, label=f'beta = {beta}', color=color, lw=1.5)

    plt.xlabel("$\hat{\omega}_*$")
    plt.ylabel("Energia média")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.show()

def main():
    # ex_02()
    # ex_04()
    ex_05()

if __name__ == "__main__":
    main()