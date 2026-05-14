import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class MembranaElastica:
    def __init__(self, N, R:float):
        self.N = N
        self.nunk = N*N

        self.R = R
        
        self.h = 2 * R / (N - 1)
        self.h_sq = self.h * self.h
        
        self.e = 0.1e-3
        self.sigma = 200
        self.rho = 900

        self.K = None
        self.M = None

        self.build_laplacian()

    def ij2n(self, i:int, j:int):
        return i + j * self.N

    def build_laplacian(self):
        N = self.N
        nunk = self.nunk

        h_sq_hat = 4.0 / ((N-1)*(N-1))

        d1 = 4.0*np.ones(nunk)
        d2 = -np.ones(nunk-1)
        d3 = -np.ones(nunk-N)
        
        K = 1.0 / h_sq_hat * sp.sparse.diags([d3, d2, d1, d2, d3], [-N, -1, 0, 1, N], format='csr')
        
        big_number = 1e7
        Iden = big_number * sp.sparse.identity(nunk, format='csr')
        
        for k in range(0,N):
            Ic = self.ij2n(0,k)
            K[Ic,:], K[:,Ic] = Iden[Ic,:], Iden[:,Ic]
            
            Ic = self.ij2n(N-1,k)
            K[Ic,:], K[:,Ic] = Iden[Ic,:], Iden[:,Ic]
        
        for k in range(0,N):
            Ic = self.ij2n(k,0)
            K[Ic,:], K[:,Ic] = Iden[Ic,:], Iden[:,Ic]
            
            Ic = self.ij2n(k,N-1)
            K[Ic,:], K[:,Ic] = Iden[Ic,:], Iden[:,Ic]
        
        h_hat = 2 / (N - 1)

        for i in range(0, N):
            for j in range(0, N):
                x = i * h_hat - 1.0
                y = j * h_hat - 1.0

                dist_squared = x*x + y*y

                if dist_squared > 1.0:
                    Ic = self.ij2n(i,j)
                    K[Ic,:], K[:,Ic] = Iden[Ic,:], Iden[:,Ic]

        M = sp.sparse.identity(nunk, format='csr')

        self.K = K
        self.M = M

        return self.K, self.M
    
    def solve_modes_adimensional(self, nmodes=10):
        assert(not(self.K is None or self.M is None))

        Lam, modes = sp.sparse.linalg.eigsh(self.K, k=nmodes, M=self.M, which='SM', tol=1e-9)

        idx = np.argsort(Lam)
        Lam = np.real(Lam[idx])
        modes = modes[:, idx]

        omegas = np.sqrt(Lam)
        freqs = omegas / (2 * np.pi)

        return freqs, omegas, modes
    
    def solve_modes(self, nmodes=10):
        freqs, omegas, modes = self.solve_modes_adimensional(nmodes)

        R = self.R
        sigma = self.sigma
        rho = self.rho
        e = self.e
        h_sq = self.h_sq

        scale = np.sqrt(sigma/(rho * e * R * R))

        omegas *= scale
        freqs *= scale

        modes /= np.sqrt(rho * e * h_sq)

        return freqs, omegas, modes

    def plot_modes(self, nmodes=10, flag_type='surface'):
        freq, _, modes = self.solve_modes(nmodes)

        for i in range(modes.shape[1]):
            mode = modes[:, i]
            f = freq[i]

            if flag_type == 'contour':
                self._plot_contour(mode, f, i+1)
            elif flag_type == 'surface':
                self._plot_surface(mode, f, i+1)

    def _plot_contour(self, mode, f, n):
        N = self.N
        R = self.R

        x = np.linspace(0.0, 2 * R, N)
        y = np.linspace(0.0, 2 * R, N)

        X, Y = np.meshgrid(x, y)
        Z = mode.copy().reshape(N, N)
        
        r = (X - R)**2 + (Y - R)**2
        Z[r > R*R] = 0
    
        fig, ax = plt.subplots(figsize=(8,4))
        
        ax.set_aspect('equal')
        ax.set(xlabel='$x$ (m)', ylabel='$y$ (m)', title=f'Modo {n} (f={f:.2f}Hz)')
        
        im = ax.contourf(X, Y, Z, 20, cmap='jet')
        im2 = ax.contour(X, Y, Z, 20, linewidths=0.25, colors='k')
        
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal')
        cbar.set_label("$w$")
            
        plt.show()

    def _plot_surface(self, mode, f, n):
        N = self.N
        R = self.R

        x = np.linspace(0.0, 2 * R, N)
        y = np.linspace(0.0, 2 * R, N)
        
        X, Y = np.meshgrid(x, y)
        Z = mode.copy().reshape(N, N)
        
        r = (X - R)**2 + (Y - R)**2
        Z[r > R*R] = 0

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8,6))
        surf = ax.plot_surface(X, Y, Z, cmap='jet')
        ax.set(xlabel='$x$ (m)', ylabel='$y$ (m)', zlabel='$w$', title=f'Modo {n} (f={f:.2f}Hz)')
        
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label("$w$")
        
        plt.show()

def ex_02():
    R = 0.4e-2
    grids = range(21, 102, 20)

    print("Tabela das frequências fundamentais da membrana (Hz)")

    print("-"*124)
    print("| Modo".ljust(13), end="|")

    for i in range(1, 11):
        print(f" {i}".ljust(10), end="|")

    print()
    print("-"*124)

    for N in grids:
        membrana = MembranaElastica(N, R)
        freqs, _, _ = membrana.solve_modes(10)

        print(f"| ({N},{N})".ljust(13), end="|")

        for f in freqs:
            print(f" {f:.2f}".ljust(10), end="|")

        print()

    print("-"*124)
    
    membrana = MembranaElastica(101, R)
    membrana.plot_modes()

def ex_04():
    R = 0.4e-2
    N = 101
    membrana = MembranaElastica(N, R)

    x_hat = np.linspace(0, 2, N)
    y_hat = np.linspace(0, 2, N)

    X, Y = np.meshgrid(x_hat, y_hat)
    Z_matrix = (X - 0.5)**2 + (Y - 0.5)**2
    
    dist_r = np.sqrt((X - 1.0)**2 + (Y - 1.0)**2)
    Z_matrix[dist_r > 1.0] = 0

    Z = Z_matrix.flatten()

    freqs_hat, omegas_hat, modes = membrana.solve_modes_adimensional(nmodes=100)
    alphas = modes.T @ Z

    omega_star_hat = 10
    beta = 0.1

    denominador = np.sqrt((omegas_hat**2 - omega_star_hat**2)**2 + (beta * omega_star_hat)**2)
    ci = alphas / denominador

    print(f"omega*^ = {omega_star_hat} rad/s e beta = {beta}")
    print("-" * 54)
    print(f"| {'Modo':<4} | {'Freq nat':<8} | {'alpha_i (Força)':<15} | {'c_i (Resposta)':<14} |")
    print("-" * 54)

    for i in range(len(freqs_hat)):
        print(f"| {i+1:<4} | {freqs_hat[i]:<8.4f} | {alphas[i]:<15.6e} | {ci[i]:<14.6e} |")
    print("-" * 54)

    return ci, alphas
    
def ex_05():
    R = 0.4e-2
    N = 71
    membrana = MembranaElastica(N, R)

    x_hat = np.linspace(0, 2, N)
    y_hat = np.linspace(0, 2, N)

    X, Y = np.meshgrid(x_hat, y_hat)
    Z_matrix = (X - 0.5)**2 + (Y - 0.5)**2
    
    dist_r = np.sqrt((X - 1.0)**2 + (Y - 1.0)**2)
    Z_matrix[dist_r > 1.0] = 0

    Z = Z_matrix.flatten()

    _, omegas_hat, modes = membrana.solve_modes_adimensional(nmodes=800)
    alphas = modes.T @ Z

    beta_values = [0.01, 0.1, 1]
    ws_hat_range = np.linspace(0.5, 100, 5000)

    plt.figure(figsize=(10, 6))
    colors = ['red', 'magenta', 'green']

    for beta, color in zip(beta_values, colors):
        Ae_list = []
        for ws_hat in ws_hat_range:
            denominador = np.sqrt((omegas_hat**2 - ws_hat**2)**2 + (beta * ws_hat)**2)
            ci = alphas / denominador
            
            Ae = 0.25 * np.sum((ci**2) * (omegas_hat**2))
            Ae_list.append(Ae)
        
        plt.loglog(ws_hat_range, Ae_list, label=f'$\\beta = {beta}$', color=color, lw=1.5)

    plt.xlabel("$\\hat{\\omega}_*$")
    plt.ylabel("Energia média")

    plt.ylim(1e-1, 1e5)

    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.show()

def main():
    # ex_02()
    ex_04()
    # ex_05()

if __name__ == "__main__":
    main()