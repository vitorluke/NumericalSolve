import numpy as np
import matplotlib.pyplot as plt

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

        self.temperaturas = np.linalg.solve(matriz_modificada, vetor_modificado)
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


if __name__ == "__main__":
    placa = PlacaTermica(Nx=21, Ny=21, condutividade=0.25, h=0.1)

    placa.resolver(T_top=100.0, T_bottom=0.0, T_left=50.0, T_right=50.0)
    
    placa.plotaPlaca(flag_type='contour')
    placa.plotaPlaca(flag_type='surface')