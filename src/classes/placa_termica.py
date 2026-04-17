from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class PlacaTermica:
    def __init__(self, Nx: int, Ny: int, condutividade: float, h: float = 1.0, fonte_calor: float = 0.0):
        """Construtor da classe da placa térmica."""
        self.Nx = Nx
        self.Ny = Ny
        self.N_total = Nx * Ny
        self.k = condutividade
        self.h = h # Espaçamento da grade
        self.fonte_calor = fonte_calor # Termo f(x,y)
        
        self.matriz_global = None
        self.vetor_fonte = None
        self.temperaturas = None

    def flatten_coordinate(self, i: int, j: int) -> int:
        """Converte as coordenadas 2D (i, j) para um índice linear 1D."""
        return i + j * self.Nx

    def assembly(self):
        """Monta a matriz global do sistema e o vetor do lado direito."""
        self.matriz_global = np.zeros((self.N_total, self.N_total))
        self.vetor_fonte = np.zeros(self.N_total)

        for i in range(self.Nx):
            for j in range(self.Ny):
                ic = self.flatten_coordinate(i, j)
                
                # Verifica se é um nó interno (fora das bordas)
                if 0 < i < self.Nx - 1 and 0 < j < self.Ny - 1:
                    ie = self.flatten_coordinate(i + 1, j)   # Leste
                    iw = self.flatten_coordinate(i - 1, j)   # Oeste
                    inorth = self.flatten_coordinate(i, j + 1) # Norte
                    isouth = self.flatten_coordinate(i, j - 1) # Sul
                    
                    # k(-Te - Tw + 4Tc - Tn - Ts) = h^2 * f
                    self.matriz_global[ic, ic] = 4.0 * self.k
                    self.matriz_global[ic, ie] = -1.0 * self.k
                    self.matriz_global[ic, iw] = -1.0 * self.k
                    self.matriz_global[ic, inorth] = -1.0 * self.k
                    self.matriz_global[ic, isouth] = -1.0 * self.k
                    
                    self.vetor_fonte[ic] = (self.h ** 2) * self.fonte_calor
                else:
                    # Condição de contorno (trivial para as bordas)
                    self.matriz_global[ic, ic] = 1.0
                    self.vetor_fonte[ic] = 0.0

    def resolver(self, T_top: float, T_bottom: float, T_left: float, T_right: float):
        """Aplica as condições de contorno e resolve o sistema linear."""
        if self.matriz_global is None:
            self.assembly()

        matriz_modificada = self.matriz_global.copy()
        vetor_modificado = self.vetor_fonte.copy()

        for i in range(self.Nx):
            for j in range(self.Ny):
                ic = self.flatten_coordinate(i, j)
                
                if i == 0:
                    matriz_modificada[ic, :] = 0
                    matriz_modificada[ic, ic] = 1.0
                    vetor_modificado[ic] = T_left
                elif i == self.Nx - 1:
                    matriz_modificada[ic, :] = 0
                    matriz_modificada[ic, ic] = 1.0
                    vetor_modificado[ic] = T_right
                elif j == 0:
                    matriz_modificada[ic, :] = 0
                    matriz_modificada[ic, ic] = 1.0
                    vetor_modificado[ic] = T_bottom
                elif j == self.Ny - 1:
                    matriz_modificada[ic, :] = 0
                    matriz_modificada[ic, ic] = 1.0
                    vetor_modificado[ic] = T_top

        matriz_esparsa = sparse.csr_matrix(matriz_modificada)
        self.temperaturas = sparse.linalg.spsolve(matriz_esparsa, vetor_modificado)

        return self.temperaturas

    def plotaPlaca(self, flag_type='contour', filename=None):
        """Plota a dispersão de calor da placa (contour 2D ou surface 3D)."""
        if self.temperaturas is None:
            print("Erro: Resolva a placa térmica antes de plotar.")
            return

        # Calculando os comprimentos físicos da placa com base nos nós e no espaçamento
        Lx = (self.Nx - 1) * self.h
        Ly = (self.Ny - 1) * self.h

        # Criando a malha de coordenadas
        x = np.linspace(0.0, Lx, self.Nx)
        y = np.linspace(0.0, Ly, self.Ny)
        X, Y = np.meshgrid(x, y)
        
        # Formatando o vetor de temperaturas (1D) para a matriz (2D) Ny x Nx
        Z = np.copy(self.temperaturas).reshape(self.Ny, self.Nx)
        
        if flag_type == 'contour':
            fig, ax = plt.subplots(figsize=(6,6))
            ax.set_aspect('equal')
            ax.set(xlabel='x', ylabel='y', title='Contours of temperature')
            im = ax.contourf(X, Y, Z, 20, cmap='jet')
            im2 = ax.contour(X, Y, Z, 20, linewidths=0.25, colors='k')
            fig.colorbar(im, ax=ax, orientation='horizontal')
            
        elif flag_type == 'surface':
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8,6))
            surf = ax.plot_surface(X, Y, Z, cmap='jet')
            ax.set(xlabel='x', ylabel='y', zlabel='Temperatura', title='Superfície de Temperatura')
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5) 
            
        plt.xticks([0, Lx/2, Lx])
        plt.yticks([0, Ly/2, Ly])

        if filename is not None:
            plt.savefig(filename)

        plt.show()
    def gerar_historico_jacobi(self, T_top: float, T_bottom: float, T_left: float, T_right: float, max_iter: int = 150):
        """Resolve a placa via Jacobi e retorna uma lista com o estado da malha a cada iteração."""
        # Inicializa a malha com zeros
        T = np.zeros((self.Nx, self.Ny))
        
        # Aplica as condições de contorno
        T[:, 0] = T_bottom
        T[:, -1] = T_top
        T[0, :] = T_left
        T[-1, :] = T_right
        
        # Dedução do termo fonte baseado na sua montagem: 4Tc - Te - Tw - Tn - Ts = (h^2 * f) / k
        termo_fonte = (self.h ** 2 * self.fonte_calor) / self.k
        historico = [T.copy()]

        for _ in range(max_iter):
            T_new = T.copy()
            # Versão vetorizada e rápida do Jacobi (calcula tudo ao mesmo tempo usando o estado antigo)
            T_new[1:-1, 1:-1] = 0.25 * (
                T[2:, 1:-1] + T[:-2, 1:-1] + # Leste e Oeste
                T[1:-1, 2:] + T[1:-1, :-2] + # Norte e Sul
                termo_fonte
            )
            T = T_new
            historico.append(T.copy())
            
        return historico

    def gerar_historico_gauss_seidel(self, T_top: float, T_bottom: float, T_left: float, T_right: float, max_iter: int = 150):
        """Resolve a placa via Gauss-Seidel e retorna uma lista com o estado da malha a cada iteração."""
        T = np.zeros((self.Nx, self.Ny))
        
        T[:, 0] = T_bottom
        T[:, -1] = T_top
        T[0, :] = T_left
        T[-1, :] = T_right
        
        termo_fonte = (self.h ** 2 * self.fonte_calor) / self.k
        historico = [T.copy()]

        for _ in range(max_iter):
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    T[i, j] = 0.25 * (
                        T[i+1, j] + T[i-1, j] + 
                        T[i, j+1] + T[i, j-1] + 
                        termo_fonte
                    )
            historico.append(T.copy())
            
        return historico

    def animar_comparacao(self, hist_jacobi, hist_gs, intervalo_ms: int = 50, filename: str = None):
        """Gera uma animação lado a lado comparando as iterações."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        Lx = (self.Nx - 1) * self.h
        Ly = (self.Ny - 1) * self.h
        X, Y = np.meshgrid(np.linspace(0.0, Lx, self.Nx), np.linspace(0.0, Ly, self.Ny))
        
        vmin = min(np.min(hist_jacobi), np.min(hist_gs))
        vmax = max(np.max(hist_jacobi), np.max(hist_gs))
        
        def update(frame):
            ax1.clear()
            ax2.clear()
            
            contorno1 = ax1.contourf(X, Y, hist_jacobi[frame].T, 20, cmap='jet', vmin=vmin, vmax=vmax)
            contorno2 = ax2.contourf(X, Y, hist_gs[frame].T, 20, cmap='jet', vmin=vmin, vmax=vmax)
            
            ax1.set_title(f"Jacobi (Iteração {frame})")
            ax2.set_title(f"Gauss-Seidel (Iteração {frame})")
            
            for ax in [ax1, ax2]:
                ax.set_aspect('equal')
                ax.set(xlabel='x', ylabel='y')
                ax.set_xticks([0, Lx/2, Lx])
                ax.set_yticks([0, Ly/2, Ly])
                
            return contorno1, contorno2

        frames_totais = min(len(hist_jacobi), len(hist_gs))
        ani = animation.FuncAnimation(fig, update, frames=frames_totais, interval=intervalo_ms, repeat=False)
        
        contorno_base = ax1.contourf(X, Y, hist_jacobi[0].T, 20, cmap='jet', vmin=vmin, vmax=vmax)
        fig.colorbar(contorno_base, ax=[ax1, ax2], orientation='horizontal', shrink=0.6, pad=0.15)
        
        if filename:
            ani.save(filename, writer='pillow')
            print(f"Animação salva como {filename}")
            
        plt.show()


if __name__ == "__main__":
    placa = PlacaTermica(Nx=21, Ny=21, condutividade=0.25, h=0.1)

    placa.resolver(T_top=100.0, T_bottom=0.0, T_left=50.0, T_right=50.0)

    historico_jac = placa.gerar_historico_jacobi(T_top=100.0, T_bottom=0.0, T_left=50.0, T_right=50.0, max_iter=200)

    historico_gs = placa.gerar_historico_gauss_seidel(T_top=100.0, T_bottom=0.0, T_left=50.0, T_right=50.0, max_iter=200)

    placa.animar_comparacao(historico_jac, historico_gs, intervalo_ms=40)
    
    placa.plotaPlaca(flag_type='contour')
    placa.plotaPlaca(flag_type='surface')