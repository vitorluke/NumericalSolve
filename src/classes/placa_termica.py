from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class PlacaTermica:
    def __init__(self, Nx: int, Ny: int, k, h: float = 1.0, fonte_calor: float = 0.0):
        """Construtor da classe da placa térmica."""
        self.Nx = Nx
        self.Ny = Ny
        self.N_total = Nx * Ny
        self.k = k
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

        if callable(self.k):
            self.assembly_variavel()
        else:
            self.assembly_uniforme()

    def assembly_uniforme(self):
        """Monta a matriz global sistema e o vetor, quando k é constante."""
        d1 =  np.ones(self.N_total) * 4.0 * self.k
        d2 = -np.ones(self.N_total - 1) * self.k
        d3 = -np.ones(self.N_total - self.Nx) * self.k

        np.fill_diagonal(self.matriz_global, d1)

        N = self.N_total

        idx = np.arange(N - 1)
        self.matriz_global[idx + 1, idx] = d2
        self.matriz_global[idx, idx + 1] = d2

        k = self.Nx
        idx_k = np.arange(N - k)
        self.matriz_global[idx_k + k, idx_k] = d3
        self.matriz_global[idx_k, idx_k + k] = d3
        
        for i in range(1, self.Nx - 1, 1):
            for j in range(1, self.Ny - 1, 1):
                ic = self.flatten_coordinate(i, j)
                self.vetor_fonte[ic] = (self.h ** 2) * self.fonte_calor

    def assembly_variavel(self):
        """Monta a matriz global sistema e o vetor, quando k é função de i e j."""
        for i in range(self.Nx):
            for j in range(self.Ny):
                ic = self.flatten_coordinate(i, j)
                
                if 0 < i < self.Nx - 1 and 0 < j < self.Ny - 1:
                    # nó interno
                    i_e = self.flatten_coordinate(i + 1, j)
                    i_w = self.flatten_coordinate(i - 1, j)
                    i_n = self.flatten_coordinate(i, j + 1)
                    i_s = self.flatten_coordinate(i, j - 1)
                    
                    k = self.k(i, j)

                    self.matriz_global[ic, ic]  =  4.0 * k

                    self.matriz_global[ic, i_e] = -1.0 * k
                    self.matriz_global[ic, i_w] = -1.0 * k
                    self.matriz_global[ic, i_n] = -1.0 * k
                    self.matriz_global[ic, i_s] = -1.0 * k
                    
                    self.vetor_fonte[ic] = (self.h ** 2) * self.fonte_calor
                else:
                    # nó da borda (termicamente isolada)
                    self.matriz_global[ic, ic] = 1.0
                    self.vetor_fonte[ic] = 0.0

    def resolver(self, boundary:list[(int, float)]):
        """Aplica as condições de contorno e resolve o sistema linear."""
        if self.matriz_global is None:
            self.assembly()

        matriz_modificada = self.matriz_global.copy()
        vetor_modificado = self.vetor_fonte.copy()

        for (ic, T) in boundary:
            matriz_modificada[ic, :]  = 0.0
            matriz_modificada[ic, ic] = 1.0

            vetor_modificado[ic]      = T

        matriz_esparsa = sparse.csr_matrix(matriz_modificada)
        self.temperaturas = sparse.linalg.spsolve(matriz_esparsa, vetor_modificado)

        return self.temperaturas
        

    def plota_placa(self, flag_type='contour', filename=None):
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

    def malha_iterativa(self, fronteira):
        malha = np.zeros((self.Nx, self.Ny))

        for (ic, T) in fronteira:
            i = ic %  self.Nx
            j = ic // self.Nx

            malha[i, j]  = T

        return malha

    def gerar_historico_jacobi(self, fronteira:list[(int, float)], max_iter: int = 150):
        """Resolve a placa via Jacobi e retorna uma lista com o estado da malha a cada iteração."""
        # Inicializa a malha com zeros
        T = self.malha_iterativa(fronteira=fronteira)
        
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

    def gerar_historico_gauss_seidel(self, fronteira:list[(int, float)], max_iter: int = 150):
        """Resolve a placa via Gauss-Seidel e retorna uma lista com o estado da malha a cada iteração."""
        T = self.malha_iterativa(fronteira=fronteira)
        
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
    
    def resolver_com_circulo(self, T_c: float, fronteira:list[(int, float)], raio: float, cx: float, cy: float):
        """
        Monta e resolve a placa aplicando uma temperatura T_c fixa 
        em uma região circular definida por um raio e centro (cx, cy).
        Retorna a matriz de temperaturas e o valor máximo encontrado.
        """
        # Pega a matriz normal com as bordas
        circulo = circulo_constante(
            T  = T_c,
            r  = raio,
            cx = cx,
            cy = cy,
            h  = self.h,
            Nx = self.Nx,
            Ny = self.Ny
        )

        fronteira_total = fronteira + circulo

        temp_1d = self.resolver(fronteira_total)
        T_max_atual = np.max(temp_1d)
        
        return temp_1d.reshape((self.Ny, self.Nx)), T_max_atual


    def descobrir_Tc_para_Tmax(self, T_alvo: float, fronteira:list[(int, float)], raio: float, cx: float, cy: float, tolerancia: float = 1e-4, max_iter: int = 50):
        """
        Usa o Método da Secante (um método iterativo simples) para encontrar 
        qual T_c faz com que T_max da placa seja exatamente T_alvo.
        """
        print(f"\n--- Iniciando busca por T_c para atingir T_max = {T_alvo}°C ---")
        
        # Chute 1
        Tc_0 = 0.0 
        _, Tmax_0 = self.resolver_com_circulo(Tc_0, fronteira, raio, cx, cy)
        erro_0 = Tmax_0 - T_alvo
        
        # Chute 2
        Tc_1 = 100.0
        _, Tmax_1 = self.resolver_com_circulo(Tc_1, fronteira, raio, cx, cy)
        erro_1 = Tmax_1 - T_alvo

        for it in range(max_iter):
            # Se a diferença entre os erros for zero, deu merda (divisão por zero)
            if abs(erro_1 - erro_0) < 1e-12:
                print("Erro: Gradiente zerou. Não foi possível convergir.")
                break
                
            # Fórmula do Método da Secante para adivinhar o próximo T_c
            Tc_novo = Tc_1 - erro_1 * ((Tc_1 - Tc_0) / (erro_1 - erro_0))
            
            # Resolve a placa com o novo T_c
            mapa_temp, Tmax_novo = self.resolver_com_circulo(Tc_novo, fronteira, raio, cx, cy)
            erro_novo = Tmax_novo - T_alvo
            
            print(f"Iteração {it+1}: Testando T_c = {Tc_novo:.4f} | T_max obtido = {Tmax_novo:.4f} | Erro = {abs(erro_novo):.6f}")
            
            # Verifica se já chegamos perto o suficiente do alvo
            if abs(erro_novo) < tolerancia:
                print(f"\n>>> SUCESSO! T_c ideal encontrado: {Tc_novo:.4f}°C")
                self.temperaturas = mapa_temp.flatten() # Salva para poder usar o plotaPlaca
                return Tc_novo
            
            # Atualiza os valores para o próximo loop
            Tc_0, erro_0 = Tc_1, erro_1
            Tc_1, erro_1 = Tc_novo, erro_novo

        print("Aviso: Limite de iterações atingido sem convergir perfeitamente.")
        return Tc_1

def fronteira_padrao(Nx, Ny):
    fronteira = []

    for j in range(Ny):
        left = (j * Nx, 10.0)
        right = (Nx - 1 + j * Nx, 30.0)

        fronteira.append(left)
        fronteira.append(right)

    for i in range(Nx):
        T = 10.0 + 20.0 * (i / (Nx - 1))

        top = (i, T)
        bottom = (i + (Ny - 1) * Nx, T)

        fronteira.append(top)
        fronteira.append(bottom)

    return fronteira

def circulo_constante(T:float, r:float, cx:float, cy:float, h:float, Nx:int, Ny:int):
    fronteira = []

    for i in range(Nx):
        for j in range(Ny):
            x = i * h
            y = j * h

            dist = (x - cx) ** 2 + (y - cy) ** 2

            if (dist <= r**2):
                fronteira.append((i + j * Nx, T))

    return fronteira

if __name__ == "__main__":
    placa = PlacaTermica(Nx=41, Ny=21, k=0.25, h=0.1)
    fronteira = fronteira_padrao(placa.Nx, placa.Ny)

    T_estrela = 150.0

    centro_x = (placa.Nx - 1) * placa.h / 2
    centro_y = (placa.Ny - 1) * placa.h / 2
    raio_circulo = 0.3

    Tc_ideal = placa.descobrir_Tc_para_Tmax(
        T_alvo=T_estrela, 
        fronteira=fronteira,
        raio=raio_circulo, cx=centro_x, cy=centro_y
    )

    placa.resolver(fronteira)

    historico_jac = placa.gerar_historico_jacobi(fronteira, max_iter=200)

    historico_gs = placa.gerar_historico_gauss_seidel(fronteira, max_iter=200)

    placa.animar_comparacao(historico_jac, historico_gs, intervalo_ms=40)
    
    placa.plota_placa(flag_type='contour')
    placa.plota_placa(flag_type='surface')