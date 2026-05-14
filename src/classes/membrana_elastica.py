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
        Nx, Ny = self.Nx, self.Ny
        nunk = self.nunk

        ds_hat = 4.0 / ((Nx-1)*(Ny-1))

        d1 = 4.0*np.ones(nunk)
        d2 = -np.ones(nunk-1)
        d3 = -np.ones(nunk-Nx)
        
        K = 1.0 / ds_hat * sp.sparse.diags([d3, d2, d1, d2, d3], [-Nx, -1, 0, 1, Nx], format='csr')
        
        big_number = 1e7
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
        
        R = self.R

        hx_hat = self.hx / R
        hy_hat = self.hy / R

        for i in range(0, Nx):
            for j in range(0, Ny):
                x = i * hx_hat - 1.0
                y = j * hy_hat - 1.0

                dist_squared = x*x + y*y

                if dist_squared > 1.0:
                    Ic = self.ij2n(i,j)
                    K[Ic,:], K[:,Ic] = Iden[Ic,:], Iden[:,Ic]

        M = sp.sparse.identity(nunk, format='csr')

        self.K = K
        self.M = M

        return self.K, self.M
    
    def solve_modes(self, nmodes=10):
        assert(not(self.K is None or self.M is None))

        Lam, modes = sp.sparse.linalg.eigsh(self.K, k=nmodes, M=self.M, which='SM')

        idx = np.argsort(Lam)
        Lam = np.real(Lam[idx])
        modes = modes[:, idx]

        R = self.R
        sigma = self.sigma
        rho = self.rho
        e = self.e

        scale = np.sqrt(sigma/(rho * e * R * R))

        omega = scale * np.sqrt(Lam)
        freq = omega / (2 * np.pi)

        return freq, omega, modes

    def plot_modes(self, nmodes=10, flag_type='surface'):
        _, _, modes = self.solve_modes(nmodes)

        for i in range(modes.shape[1]):
            mode = modes[:, i]

            if flag_type == 'contour':
                self._plot_contour(mode)
            elif flag_type == 'surface':
                self._plot_surface(mode)

    def _plot_contour(self, mode):
        Nx, Ny = self.Nx, self.Ny
        R = self.R

        x = np.linspace(0.0, 2 * R, Nx)
        y = np.linspace(0.0, 2 * R, Ny)

        X, Y = np.meshgrid(x, y)
        Z = mode.copy().reshape(Ny, Nx)
        
        r = (X - 0.004)**2 + (Y - 0.004)**2
        Z[r > R*R] = 0
    
        fig, ax = plt.subplots(figsize=(8,4))
        
        ax.set_aspect('equal')
        ax.set(xlabel='$x$ (m)', ylabel='$y$ (m)')
        
        im = ax.contourf(X, Y, Z, 20, cmap='jet')
        im2 = ax.contour(X, Y, Z, 20, linewidths=0.25, colors='k')
        
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal')
        cbar.set_label("Deslocamento vertical $w$ (m)")
            
        plt.show()

    def _plot_surface(self, mode):
        Nx, Ny = self.Nx, self.Ny
        R = self.R

        x = np.linspace(0.0, 2 * R, Nx)
        y = np.linspace(0.0, 2 * R, Ny)
        
        X, Y = np.meshgrid(x, y)
        Z = mode.copy().reshape(Ny, Nx)
        
        r = (X - 0.004)**2 + (Y - 0.004)**2
        Z[r > R*R] = 0

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8,6))
        surf = ax.plot_surface(X, Y, Z, cmap='jet')
        ax.set(xlabel='$x$ (m)', ylabel='$y$ (m)', zlabel='$w$ (m)')
        
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label("Deslocamento vertical $w$ (m)")
        
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
    ex_02()
    # ex_04()
    # ex_05()

if __name__ == "__main__":
    main()