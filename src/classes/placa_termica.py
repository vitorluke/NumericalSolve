from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
from numba import njit

class PlacaTermica:
    def __init__(self, Lx:float, Ly:float, Nx: int, Ny: int, k, fonte_calor: float = 0.0):
        """Construtor da classe da placa térmica."""
        self.Lx = Lx
        self.Ly = Ly
        
        self.Nx = Nx
        self.Ny = Ny
        self.N_total = Nx * Ny
 
        self.hx = Lx / (Nx - 1)
        self.hy = Ly / (Ny - 1)

        self.ds = self.hx * self.hy

        self.k = k

        self.fonte_calor = fonte_calor

        self.A = None
        self.b = None        
        self.T = None

        self.tempos_execucao = {}

    def flatten_coordinate(self, i: int, j: int) -> int:
        return i + j * self.Nx

    def _montar_densa(self, fronteira):
        tempo_inicio = time.time()

        self.A = np.zeros((self.N_total, self.N_total))
        self.b = np.zeros(self.N_total)

        if callable(self.k):
            self._montar_densa_variavel()
        else:
            self._montar_densa_uniforme()

        for (ic, t) in fronteira:
            self.A[ic, :]  = 0.0
            self.A[ic, ic] = 1.0

            self.b[ic] = t

        tempo_fim = time.time()

        self.tempos_execucao['montagem'] = tempo_fim - tempo_inicio

    def _montar_densa_uniforme(self):
        d1 =  np.ones(self.N_total) * 4.0 * self.k
        d2 = -np.ones(self.N_total - 1) * self.k
        d3 = -np.ones(self.N_total - self.Nx) * self.k

        np.fill_diagonal(self.A, d1)

        N = self.N_total

        idx = np.arange(N - 1)
        self.A[idx + 1, idx] = d2
        self.A[idx, idx + 1] = d2

        k = self.Nx
        idx_k = np.arange(N - k)
        self.A[idx_k + k, idx_k] = d3
        self.A[idx_k, idx_k + k] = d3
        
        for i in range(1, self.Nx - 1, 1):
            for j in range(1, self.Ny - 1, 1):
                ic = self.flatten_coordinate(i, j)
                self.b[ic] = self.ds * self.fonte_calor

    def _montar_densa_variavel(self):        
        for i in range(1, self.Nx - 1):
            for j in range(1, self.Ny - 1):
                ic = self.flatten_coordinate(i, j)
                
                i_e = self.flatten_coordinate(i + 1, j)
                i_w = self.flatten_coordinate(i - 1, j)
                i_n = self.flatten_coordinate(i, j + 1)
                i_s = self.flatten_coordinate(i, j - 1)
                    
                x = i * self.hx
                y = j * self.hy

                k_e = self.k(x + self.hx / 2, y)
                k_w = self.k(x - self.hx / 2, y)
                k_n = self.k(x, y + self.hy / 2)
                k_s = self.k(x, y - self.hy / 2)

                coef_central = k_e + k_w + k_n + k_s

                self.A[ic, ic]  =  coef_central

                self.A[ic, i_e] = -k_e
                self.A[ic, i_w] = -k_w
                self.A[ic, i_n] = -k_n
                self.A[ic, i_s] = -k_s
                    
                self.b[ic] = (self.ds) * self.fonte_calor

    def _resolver_densa(self, fronteira:list[(int, float)]):
        self._montar_densa(fronteira)

        tempo_inicio = time.time()

        self.T = np.linalg.solve(self.A, self.b)

        tempo_fim = time.time()

        self.tempos_execucao['resolucao'] = tempo_fim - tempo_inicio
        self.tempos_execucao['total'] = self.tempos_execucao['montagem'] + self.tempos_execucao['resolucao']

        return self.T
    
    def _montar_esparsa(self, fronteira:list[(int, float)]):
        tempo_inicio = time.time()

        rows = []
        cols = []
        data = []

        self.b = np.zeros(self.N_total)

        for (ic, t) in fronteira:
            rows.append(ic)
            cols.append(ic)
            data.append(1.0)

            self.b[ic] = t

        if (callable(self.k)):
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    ic = self.flatten_coordinate(i, j)

                    if any(r == ic for r, _ in fronteira):
                        continue

                    i_e = self.flatten_coordinate(i + 1, j)
                    i_w = self.flatten_coordinate(i - 1, j)
                    i_n = self.flatten_coordinate(i, j + 1)
                    i_s = self.flatten_coordinate(i, j - 1)
                    
                    x = i * self.hx
                    y = j * self.hy

                    k_e = self.k(x + self.hx / 2, y)
                    k_w = self.k(x - self.hx / 2, y)
                    k_n = self.k(x, y + self.hy / 2)
                    k_s = self.k(x, y - self.hy / 2)

                    coef_central = k_e + k_w + k_n + k_s

                    rows.extend([ic, ic, ic, ic, ic])
                    cols.extend([ic, i_e, i_w, i_n, i_s])
                    data.extend([coef_central, -k_e, -k_w, -k_n, -k_s])

                    self.b[ic] = (self.ds) * self.fonte_calor
        else:
            k = self.k

            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    ic = self.flatten_coordinate(i, j)

                    if any(r == ic for r, _ in fronteira):
                        continue

                    i_e = self.flatten_coordinate(i + 1, j)
                    i_w = self.flatten_coordinate(i - 1, j)
                    i_n = self.flatten_coordinate(i, j + 1)
                    i_s = self.flatten_coordinate(i, j - 1)
                    
                    rows.extend([ic, ic, ic, ic, ic])
                    cols.extend([ic, i_e, i_w, i_n, i_s])
                    data.extend([4.0 * k, -k, -k, -k, -k])

                    self.b[ic] = (self.ds) * self.fonte_calor

        self.A = sparse.csr_matrix((data, (rows, cols)), shape=(self.N_total, self.N_total))

        tempo_fim = time.time()

        self.tempos_execucao['montagem'] = tempo_fim - tempo_inicio

        return self.T

    def _resolver_esparsa(self, fronteira:list[(int, float)]):
        self._montar_esparsa(fronteira)

        tempo_inicio = time.time()

        self.T = sparse.linalg.spsolve(self.A, self.b)

        tempo_fim = time.time()

        self.tempos_execucao['resolucao'] = tempo_fim - tempo_inicio
        self.tempos_execucao['total'] = self.tempos_execucao['montagem'] + self.tempos_execucao['resolucao']

        return self.T
    
    def resolver(self, fronteira:list[(int, float)], use_sparse:bool=True):
        if use_sparse:
            return self._resolver_esparsa(fronteira)
        else:
            return self._resolver_densa(fronteira)

    def temp_max(self):
        if self.T is None:
            return None
        else:
            return np.max(self.T)

    def temp_med(self):
        if self.T is None:
            return None
        else:
            return np.average(self.T)

    def plota_placa(self, flag_type='contour', title='Distribuição de temperatura', filename=None):
        """Plota a dispersão de calor da placa (contour 2D ou surface 3D)."""
        if self.T is None:
            print("Erro: Resolva a placa antes de plotar.")
            return

        x = np.linspace(0.0, self.Lx, self.Nx)
        y = np.linspace(0.0, self.Ly, self.Ny)
        X, Y = np.meshgrid(x, y)
        Z = np.copy(self.T).reshape(self.Ny, self.Nx)
        
        if flag_type == 'contour':
            fig, ax = plt.subplots(figsize=(8,4))
            ax.set_aspect('equal')
            ax.set(xlabel='x (m)', ylabel='y (m)', title=title)
            im = ax.contourf(X, Y, Z, 20, cmap='jet')
            im2 = ax.contour(X, Y, Z, 20, linewidths=0.25, colors='k')
            fig.colorbar(im, ax=ax, orientation='horizontal')
        elif flag_type == 'surface':
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8,6))
            surf = ax.plot_surface(X, Y, Z, cmap='jet')
            ax.set(xlabel='x', ylabel='y', zlabel='Temperatura (°C)', title=title)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5) 
            
        plt.xticks([0, self.Lx/2, self.Lx])
        plt.yticks([0, self.Ly/2, self.Ly])

        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def _malha_iterativa(self, fronteira):
        tempo_inicio = time.time()

        T = np.zeros((self.Nx, self.Ny))

        for (ic, t) in fronteira:
            i = ic %  self.Nx
            j = ic // self.Nx

            T[i, j] = t

        tempo_fim = time.time()
        self.tempos_execucao['montagem'] = tempo_fim - tempo_inicio

        return T

    def _chute_inicial(self, borda):
        T = np.zeros(self.N_total)

        for (ic, t) in borda:
            T[ic] = t

        return T

    def gerar_historico_jacobi(self, fronteira:list[(int, float)], max_iter:int=1000, frame_skip:int=10):
        """Resolve a placa via Jacobi e retorna uma lista com o estado da malha a cada iteração."""
        # Inicializa a malha com zeros
        T = self._malha_iterativa(fronteira=fronteira)
        
        # Dedução do termo fonte baseado na sua montagem: 4Tc - Te - Tw - Tn - Ts = (h^2 * f) / k
        ke, kw, kn, ks, den = self._obter_vizinhos()

        termo_fonte = self.ds * self.fonte_calor
        historico = [(0, T.copy())]

        for n in range(1, max_iter + 1):
            T_new = T.copy()

            T_new[1:-1, 1:-1] = (
                ke*T[2:, 1:-1] + kw*T[:-2, 1:-1] +
                kn*T[1:-1, 2:] + ks*T[1:-1, :-2] +
                termo_fonte
            ) / den

            T[:] = T_new

            if (n % frame_skip == 0):
                historico.append((n, T.copy()))
            
        return historico

    def gerar_historico_gauss_seidel(self, borda:list[(int, float)], max_iter:int=1000, omega=1.85, frame_skip:int=10):
        """Resolve a placa via Gauss-Seidel e retorna uma lista com o estado da malha a cada iteração."""
        T = self._malha_iterativa(borda)
        
        ke, kw, kn, ks, den = self._obter_vizinhos()
        termo_fonte = self.ds * self.fonte_calor

        historico = [(0, T.copy())]

        for n in range(1, max_iter + 1):
            max_diff = 0.0

            Tgs = (
                ke[::2, ::2]*T[2::2,1:-1:2] +
                kw[::2, ::2]*T[:-2:2,1:-1:2] +
                kn[::2, ::2]*T[1:-1:2,2::2] +
                ks[::2, ::2]*T[1:-1:2,:-2:2] +
                termo_fonte
            ) / den[::2, ::2]

            old = T[1:-1:2, 1:-1:2]
            new = (1 - omega)*old + omega*Tgs
            max_diff = max(max_diff, np.max(np.abs(new - old)))
            T[1:-1:2, 1:-1:2] = new

            Tgs = (
                ke[1::2,1::2]*T[3::2,2:-1:2] +
                kw[1::2,1::2]*T[1:-2:2,2:-1:2] +
                kn[1::2,1::2]*T[2:-1:2,3::2] +
                ks[1::2,1::2]*T[2:-1:2,1:-2:2] +
                termo_fonte
            ) / den[1::2,1::2]

            old = T[2:-1:2, 2:-1:2]
            new = (1 - omega)*old + omega*Tgs
            max_diff = max(max_diff, np.max(np.abs(new - old)))
            T[2:-1:2, 2:-1:2] = new

            Tgs = (
                ke[::2,1::2]*T[2::2,2:-1:2] +
                kw[::2,1::2]*T[:-2:2,2:-1:2] +
                kn[::2,1::2]*T[1:-1:2,3::2] +
                ks[::2,1::2]*T[1:-1:2,1:-2:2] +
                termo_fonte
            ) / den[::2,1::2]

            old = T[1:-1:2, 2:-1:2]
            new = (1 - omega)*old + omega*Tgs
            max_diff = max(max_diff, np.max(np.abs(new - old)))
            T[1:-1:2, 2:-1:2] = new

            Tgs = (
                ke[1::2,::2]*T[3::2,1:-1:2] +
                kw[1::2,::2]*T[1:-2:2,1:-1:2] +
                kn[1::2,::2]*T[2:-1:2,2::2] +
                ks[1::2,::2]*T[2:-1:2,:-2:2] +
                termo_fonte
            ) / den[1::2,::2]

            old = T[2:-1:2, 1:-1:2]
            new = (1 - omega)*old + omega*Tgs
            max_diff = max(max_diff, np.max(np.abs(new - old)))
            T[2:-1:2, 1:-1:2] = new

            if (n % frame_skip == 0):
                historico.append((n, T.copy()))

        return historico

    def animar_comparacao(self, hist_jacobi, hist_gs, intervalo_ms: int = 50, filename: str = None):
        """Gera uma animação lado a lado comparando as iterações."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        Lx = self.Lx
        Ly = self.Ly
        X, Y = np.meshgrid(np.linspace(0.0, Lx, self.Nx), np.linspace(0.0, Ly, self.Ny))
        
        h_jc = list(map(lambda x: x[1], hist_jacobi))
        h_gs = list(map(lambda x: x[1], hist_gs))

        vmin = min(np.min(h_jc), np.min(h_gs))
        vmax = max(np.max(h_jc), np.max(h_gs))
        
        def update(frame):
            ax1.clear()
            ax2.clear()

            jacobi = hist_jacobi[frame]
            gs     = hist_gs[frame]
            
            contorno1 = ax1.contourf(X, Y, jacobi[1].T, 20, cmap='jet', vmin=vmin, vmax=vmax)
            contorno2 = ax2.contourf(X, Y, gs[1].T, 20, cmap='jet', vmin=vmin, vmax=vmax)
            
            ax1.set_title(f"Jacobi (Iteração {jacobi[0]})")
            ax2.set_title(f"Gauss-Seidel (Iteração {gs[0]})")
            
            for ax in [ax1, ax2]:
                ax.set_aspect('equal')
                ax.set(xlabel='x', ylabel='y')
                ax.set_xticks([0, Lx/2, Lx])
                ax.set_yticks([0, Ly/2, Ly])
                
            return contorno1, contorno2

        frames_totais = min(len(hist_jacobi), len(hist_gs))
        ani = animation.FuncAnimation(fig, update, frames=frames_totais, interval=intervalo_ms, repeat=False)
        
        contorno_base = ax1.contourf(X, Y, h_jc[-1].T, 20, cmap='jet', vmin=vmin, vmax=vmax)
        fig.colorbar(contorno_base, ax=[ax1, ax2], orientation='horizontal', shrink=0.6, pad=0.15)
        
        if filename:
            ani.save(filename, writer='pillow')
            print(f"Animação salva como {filename}")
            
        plt.show()
    
    def _resolver_com_circulo(self, T_c: float, fronteira:list[(int, float)], raio: float, cx: float, cy: float):
        """
        Monta e resolve a placa aplicando uma temperatura T_c fixa 
        em uma região circular definida por um raio e centro (cx, cy).
        Retorna a matriz de temperaturas e o valor máximo encontrado.
        """
        
        circulo = circulo_constante(
            T  = T_c,
            r  = raio,
            cx = cx,
            cy = cy,
            Nx = self.Nx,
            Ny = self.Ny,
            hx = self.hx,
            hy = self.hy
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
        _, Tmax_0 = self._resolver_com_circulo(Tc_0, fronteira, raio, cx, cy)
        erro_0 = Tmax_0 - T_alvo
        
        # Chute 2
        Tc_1 = 100.0
        _, Tmax_1 = self._resolver_com_circulo(Tc_1, fronteira, raio, cx, cy)
        erro_1 = Tmax_1 - T_alvo

        for it in range(max_iter):
            # Se a diferença entre os erros for zero, deu merda (divisão por zero)
            if abs(erro_1 - erro_0) < 1e-12:
                print("Erro: Gradiente zerou. Não foi possível convergir.")
                break
                
            # Fórmula do Método da Secante para adivinhar o próximo T_c
            Tc_novo = Tc_1 - erro_1 * ((Tc_1 - Tc_0) / (erro_1 - erro_0))
            
            # Resolve a placa com o novo T_c
            mapa_temp, Tmax_novo = self._resolver_com_circulo(Tc_novo, fronteira, raio, cx, cy)
            erro_novo = Tmax_novo - T_alvo
            
            print(f"Iteração {it+1}: Testando T_c = {Tc_novo:.4f} | T_max obtido = {Tmax_novo:.4f} | Erro = {abs(erro_novo):.6f}")
            
            # Verifica se já chegamos perto o suficiente do alvo
            if abs(erro_novo) < tolerancia:
                print(f"\n>>> SUCESSO! T_c ideal encontrado: {Tc_novo:.4f}°C")
                self.T = mapa_temp.flatten() # Salva para poder usar o plota_placa
                return Tc_novo
            
            # Atualiza os valores para o próximo loop
            Tc_0, erro_0 = Tc_1, erro_1
            Tc_1, erro_1 = Tc_novo, erro_novo

        print("Aviso: Limite de iterações atingido sem convergir perfeitamente.")
        return Tc_1
    
    def get_central_profile(self):
        """Retorna o perfil de temperaturas ao longo do eixo X central (y = Ly/2)."""
        j_center = self.Ny // 2
        Z = np.copy(self.T).reshape(self.Ny, self.Nx)
        x = np.linspace(0.0, self.Lx, self.Nx)
        return x, Z[j_center, :]
    
    def plota_eixo_central(self, title:str):
        x_perfil, T_perfil = self.get_central_profile()
        plt.figure(figsize=(6,4))
        plt.plot(x_perfil, T_perfil, 'b-', label='Temperatura em y = Ly/2')
        plt.title(title)
        plt.xlabel("x (m)")
        plt.ylabel("Temperatura (°C)")
        plt.grid(True)
        plt.legend()
        plt.show()

    def _obter_vizinhos(self):
        k = None

        if callable(self.k):
            x = np.linspace(0, self.Lx, self.Nx)
            y = np.linspace(0, self.Ly, self.Ny)

            X, Y = np.meshgrid(x, y, indexing="ij")

            k = self.k(X, Y)
        else:
            k = self.k * np.ones((self.Nx, self.Ny))

        kc = k[1:-1, 1:-1]

        ke = 2*kc*k[2:,1:-1] / (kc + k[2:,1:-1])
        kw = 2*kc*k[:-2,1:-1] / (kc + k[:-2,1:-1])
        kn = 2*kc*k[1:-1,2:] / (kc + k[1:-1,2:])
        ks = 2*kc*k[1:-1,:-2] / (kc + k[1:-1,:-2])

        den = ke + kw + kn + ks

        return ke, kw, kn, ks, den

    def _res_jacobi_uniforme(self, borda, tol, max_iter):
        T = self._malha_iterativa(borda)

        tempo_inicio = time.time()
        termo_fonte = (self.ds * self.fonte_calor) / self.k

        sucesso = False

        for _ in range(max_iter):
            T_new = T.copy()
            T_new[1:-1, 1:-1] = 0.25 * (
                T[2:, 1:-1] + T[:-2, 1:-1] +
                T[1:-1, 2:] + T[1:-1, :-2] +
                termo_fonte
            )

            diff = np.max(T_new - T)

            T[:] = T_new

            if diff < tol:
                sucesso = True
                break

        self.T = T.T.reshape(self.N_total)

        tempo_fim = time.time()
        self.tempos_execucao['resolucao'] = tempo_fim - tempo_inicio
        self.tempos_execucao['total'] = self.tempos_execucao['montagem'] + self.tempos_execucao['resolucao']

        if not sucesso:
            print("Número máximo de iterações atingido (Jacobi).")

        return self.T
    
    def _res_jacobi_variavel(self, borda, tol, max_iter):
        T = self._malha_iterativa(borda)

        tempo_inicio = time.time()
        sucesso = False

        ke, kw, kn, ks, den = self._obter_vizinhos()

        termo_fonte = self.ds * self.fonte_calor

        for _ in range(max_iter):
            T_new = T.copy()

            T_new[1:-1, 1:-1] = (
                ke*T[2:, 1:-1] + kw*T[:-2, 1:-1] +
                kn*T[1:-1, 2:] + ks*T[1:-1, :-2] +
                termo_fonte
            ) / den

            diff = np.max(T_new - T)

            T[:] = T_new

            if diff < tol:
                sucesso = True
                break

        self.T = T.T.reshape(self.N_total)

        tempo_fim = time.time()
        self.tempos_execucao['resolucao'] = tempo_fim - tempo_inicio
        self.tempos_execucao['total'] = self.tempos_execucao['montagem'] + self.tempos_execucao['resolucao']

        if not sucesso:
            print("Número máximo de iterações atingido (Jacobi).")

        return self.T

    def resolver_jacobi(self, borda, tol, max_iter=1000):
        if callable(self.k):
            return self._res_jacobi_variavel(borda, tol, max_iter)
        else:
            return self._res_jacobi_uniforme(borda, tol, max_iter)

    def _res_gs_rb_uniforme(self, borda, tol, max_iter, omega):
        T = self._malha_iterativa(borda)

        tempo_inicio = time.time()
        sucesso = False
        termo_fonte = (self.ds * self.fonte_calor) / self.k

        for _ in range(max_iter):
            T_old = T.copy()

            Tgs = 0.25 * (
                T[2::2, 1:-1:2] + T[:-2:2, 1:-1:2] +
                T[1:-1:2, 2::2] + T[1:-1:2, :-2:2] +
                termo_fonte
            )
            T[1:-1:2, 1:-1:2] = (1 - omega) * T[1:-1:2, 1:-1:2] + omega * Tgs

            Tgs = 0.25 * (
                T[3::2, 2:-1:2] + T[1:-2:2, 2:-1:2] +
                T[2:-1:2, 3::2] + T[2:-1:2, 1:-2:2] +
                termo_fonte
            )
            T[2:-1:2, 2:-1:2] = (1 - omega) * T[2:-1:2, 2:-1:2] + omega * Tgs

            Tgs = 0.25 * (
                T[2::2, 2:-1:2] + T[:-2:2, 2:-1:2] +
                T[1:-1:2, 3::2] + T[1:-1:2, 1:-2:2] +
                termo_fonte
            )
            T[1:-1:2, 2:-1:2] = (1 - omega) * T[1:-1:2, 2:-1:2] + omega * Tgs

            Tgs = 0.25 * (
                T[3::2, 1:-1:2] + T[1:-2:2, 1:-1:2] +
                T[2:-1:2, 2::2] + T[2:-1:2, :-2:2] +
                termo_fonte
            )
            T[2:-1:2, 1:-1:2] = (1 - omega) * T[2:-1:2, 1:-1:2] + omega * Tgs

            if np.max(T - T_old) < tol:
                sucesso = True
                break

        self.T = T.T.reshape(self.N_total)

        tempo_fim = time.time()
        self.tempos_execucao['resolucao'] = tempo_fim - tempo_inicio
        self.tempos_execucao['total'] = self.tempos_execucao['montagem'] + self.tempos_execucao['resolucao']

        if not sucesso:
            print("Número máximo de iterações atingido (Gauss-Seidel).")
                
        return self.T
    
    def _res_gs_rb_variavel(self, borda, tol, max_iter, omega):
        T = self._malha_iterativa(borda)

        tempo_inicio = time.time()
        sucesso = False
        
        ke, kw, kn, ks, den = self._obter_vizinhos()
        termo_fonte = self.ds * self.fonte_calor

        for _ in range(max_iter):
            max_diff = 0.0

            Tgs = (
                ke[::2, ::2]*T[2::2,1:-1:2] +
                kw[::2, ::2]*T[:-2:2,1:-1:2] +
                kn[::2, ::2]*T[1:-1:2,2::2] +
                ks[::2, ::2]*T[1:-1:2,:-2:2] +
                termo_fonte
            ) / den[::2, ::2]

            old = T[1:-1:2, 1:-1:2]
            new = (1 - omega)*old + omega*Tgs
            max_diff = max(max_diff, np.max(np.abs(new - old)))
            T[1:-1:2, 1:-1:2] = new

            Tgs = (
                ke[1::2,1::2]*T[3::2,2:-1:2] +
                kw[1::2,1::2]*T[1:-2:2,2:-1:2] +
                kn[1::2,1::2]*T[2:-1:2,3::2] +
                ks[1::2,1::2]*T[2:-1:2,1:-2:2] +
                termo_fonte
            ) / den[1::2,1::2]

            old = T[2:-1:2, 2:-1:2]
            new = (1 - omega)*old + omega*Tgs
            max_diff = max(max_diff, np.max(np.abs(new - old)))
            T[2:-1:2, 2:-1:2] = new

            Tgs = (
                ke[::2,1::2]*T[2::2,2:-1:2] +
                kw[::2,1::2]*T[:-2:2,2:-1:2] +
                kn[::2,1::2]*T[1:-1:2,3::2] +
                ks[::2,1::2]*T[1:-1:2,1:-2:2] +
                termo_fonte
            ) / den[::2,1::2]

            old = T[1:-1:2, 2:-1:2]
            new = (1 - omega)*old + omega*Tgs
            max_diff = max(max_diff, np.max(np.abs(new - old)))
            T[1:-1:2, 2:-1:2] = new

            Tgs = (
                ke[1::2,::2]*T[3::2,1:-1:2] +
                kw[1::2,::2]*T[1:-2:2,1:-1:2] +
                kn[1::2,::2]*T[2:-1:2,2::2] +
                ks[1::2,::2]*T[2:-1:2,:-2:2] +
                termo_fonte
            ) / den[1::2,::2]

            old = T[2:-1:2, 1:-1:2]
            new = (1 - omega)*old + omega*Tgs
            max_diff = max(max_diff, np.max(np.abs(new - old)))
            T[2:-1:2, 1:-1:2] = new

            if max_diff < tol:
                sucesso = True
                break

        self.T = T.T.reshape(self.N_total)

        tempo_fim = time.time()
        self.tempos_execucao['resolucao'] = tempo_fim - tempo_inicio
        self.tempos_execucao['total'] = self.tempos_execucao['montagem'] + self.tempos_execucao['resolucao']

        if not sucesso:
            print("Número máximo de iterações atingido (Gauss-Seidel).")
                
        return self.T

    def resolver_gauss_seidel_rb(self, borda, tol, max_iter=1000, omega=1.85):
        if callable(self.k):
            return self._res_gs_rb_variavel(borda, tol, max_iter, omega)
        else:
            return self._res_gs_rb_uniforme(borda, tol, max_iter, omega)

    def resolver_gauss_seidel(self, borda, tol, max_iter=1000, omega=1.85):
        self._montar_esparsa(borda)

        tempo_inicio = time.time()

        self.T = self._chute_inicial(borda)

        D = sp.diags(self.A.diagonal())
        L = sp.tril(self.A, k=-1)
        U = sp.triu(self.A, k=1)

        DL = (D + omega * L).tocsc()
        UR = (omega * U + (omega - 1.0) * D).tocsr()

        sucesso = False

        del D, L, U

        for _ in range(max_iter):
            T_new = spla.spsolve_triangular(DL, omega * self.b - UR @ self.T, lower=True)

            diff = np.max(T_new - self.T)

            self.T[:] = T_new

            if diff < tol:
                sucesso = True
                break

        tempo_fim = time.time()
        self.tempos_execucao['resolucao'] = tempo_fim - tempo_inicio
        self.tempos_execucao['total'] = self.tempos_execucao['montagem'] + self.tempos_execucao['resolucao']

        if not sucesso:
            print("Número máximo de iterações atingido (Gauss-Seidel).")

        return self.T

def borda_padrao(Nx:int, Ny:int):
    borda = []

    for j in range(Ny):
        left = (j * Nx, 10.0)
        right = (Nx - 1 + j * Nx, 30.0)

        borda.append(left)
        borda.append(right)

    for i in range(1, Nx - 1):
        T = 10.0 + 20.0 * (i / (Nx - 1))

        top = (i, T)
        bottom = (i + (Ny - 1) * Nx, T)

        borda.append(top)
        borda.append(bottom)

    return borda

def borda_constante(Nx:int, Ny:int, T:float):
    borda = []

    for j in range(Ny):
        left = (j * Nx, T)
        right = (Nx - 1 + j * Nx, T)

        borda.append(left)
        borda.append(right)

    for i in range(1, Nx - 1):
        top = (i, T)
        bottom = (i + (Ny - 1) * Nx, T)

        borda.append(top)
        borda.append(bottom)

    return borda

def circulo_constante(T:float, r:float, cx:float, cy:float, Nx:int, Ny:int, hx:float, hy:float):
    circulo = []

    r_squared = r ** 2

    for i in range(Nx):
        for j in range(Ny):
            x = i * hx
            y = j * hy

            square_dist = (x - cx) ** 2 + (y - cy) ** 2

            if (square_dist <= r_squared):
                circulo.append((i + j * Nx, T))

    return circulo

# --- PARÂMETROS BASE SUGERIDOS ---
Lx, Ly = 0.02, 0.01  # 2 cm x 1 cm em metros
k_nominal = 0.2
fonte_calor_nominal = 5e5

raio_circulo = 0.002
x_circulo = 0.75 * Lx
y_circulo = 0.5 * Ly

def exercicio_1():
    print("\n--- EXERCÍCIO 1 ---")
    malhas = [(21, 11), (41, 21), (81, 41), (161, 81), (321, 161), (641, 321)]
    
    print(f"{'Malha':<12} | {'T_max (°C)':<11} | {'Tempo Esparsa (s)':<18} | {'Tempo Densa (s)':<15}")
    print("-" * 65)
    
    for Nx, Ny in malhas:
        placa = PlacaTermica(Lx, Ly, Nx, Ny, k=k_nominal, fonte_calor=fonte_calor_nominal)
        temp_bordas = borda_padrao(Nx, Ny)
        
        # Teste com Matriz Esparsa
        placa.resolver(temp_bordas, use_sparse=True)
        tempo_esparsa = placa.tempos_execucao['total']
        t_max = placa.temp_max()
        
        # Teste com Matriz Densa (CUIDADO com falta de memória para malhas grandes)
        tempo_densa = "N/A (Memória)"
        if Nx * Ny < 50000: # Proteção contra MemoryError
            placa.resolver(temp_bordas, use_sparse=False)
            tempo_densa = f"{placa.tempos_execucao['total']:.5f}"
        
        print(f"({Nx}, {Ny})".ljust(11), f" | {t_max:.2f}".ljust(13), f" | {tempo_esparsa:.5f}".ljust(20), f" | {tempo_densa}")
        
        # Plotar apenas para uma malha intermediária para não abrir 50 janelas
        if Nx == 641:
            placa.plota_placa(title=f"Distribuição de temperatura (°C) ({Nx}x{Ny})")
            placa.plota_eixo_central("Perfil de temperaturas no eixo central")

def exercicio_2(T_c:float=30.0):
    print("\n--- EXERCÍCIO 2 ---")
    malhas = [(41, 21), (81, 41), (161, 81)]

    for Nx, Ny in malhas:
        placa = PlacaTermica(
            Lx,
            Ly,
            Nx,
            Ny,
            k=k_nominal,
            fonte_calor=fonte_calor_nominal
        )
        
        temp_bordas = borda_padrao(Nx, Ny)

        circulo = circulo_constante(
            T  = T_c,
            r  = raio_circulo,
            cx = x_circulo,
            cy = y_circulo,
            Nx = placa.Nx,
            Ny = placa.Ny,
            hx = placa.hx,
            hy = placa.hy
        )

        fronteira_total = circulo + temp_bordas

        placa.resolver(fronteira_total)
        
        print(f"Malha {Nx}x{Ny} -> T_max = {placa.temp_max():.2f}")
        if Nx == 161:
            placa.plota_placa(title=f"Distribuição de temperaturas (°C) com círculo constante ({Nx}x{Ny})")
            placa.plota_eixo_central("Perfil de temperaturas no eixo central com círculo constante")

def exercicio_3():
    print("\n--- EXERCÍCIO 3 ---")
    
    def k_variavel(x, y):
        return 0.2 + 0.05 * np.sin(3*np.pi*x / Lx) * np.sin(3*np.pi*y / Ly)

    Nx = 161
    Ny = 81

    placa = PlacaTermica(Lx, Ly, Nx, Ny, k=k_variavel, fonte_calor=fonte_calor_nominal)

    temp_bordas = borda_padrao(Nx, Ny)

    # placa.resolver_gauss_seidel_rb(temp_bordas, tol=1e-12, max_iter=10000, omega=1.85)
    # placa.resolver_jacobi(temp_bordas, tol=1e-12, max_iter=100000)
    placa.resolver(temp_bordas)
    
    print(f"Temperatura Máxima com k variável: {placa.temp_max():.2f}")
    placa.plota_placa(title="Distribuição de temperaturas (°C) com k variável")
    placa.plota_eixo_central("Eixo central de temperaturas (k variável)")

def exercicio_4():
    print("\n--- EXERCÍCIO 4 ---")
    Nx, Ny = 101, 51
    temperaturas_Tc = np.linspace(0, 100, 10)
    maximas = []
    medias = []
    
    for Tc in temperaturas_Tc:
        placa = PlacaTermica(Lx, Ly, Nx, Ny, k=k_nominal, fonte_calor=fonte_calor_nominal)

        temp_bordas = borda_padrao(Nx, Ny)

        circulo = circulo_constante(
            T  = Tc,
            r  = raio_circulo,
            cx = x_circulo,
            cy = y_circulo,
            Nx = placa.Nx,
            Ny = placa.Ny,
            hx = placa.hx,
            hy = placa.hy
        )

        fronteira_total = circulo + temp_bordas

        placa.resolver(fronteira_total)

        maximas.append(placa.temp_max())
        medias.append(placa.temp_med())
        
    plt.figure(figsize=(6,4))
    plt.plot(temperaturas_Tc, maximas, 'r-o', label='T_max')
    plt.plot(temperaturas_Tc, medias, 'b-o', label='T_media')
    plt.xlabel('Temperatura T_c no Cilindro (°C)')
    plt.ylabel('Temperatura (°C)')
    plt.title('Temperaturas máximas e médias por T_c')
    plt.grid()
    plt.legend()
    plt.show()

def exercicio_5():
    print("\n--- EXERCÍCIO 5 ---")
    # Para encontrar 3 coeficientes (a,b,c), precisamos de 3 "experimentos"
    # TR representa a temperatura das bordas (vamos assumir todas iguais para simplificar)
    Nx, Ny = 81, 41
    
    ic_escolhido = 233 # Índice do nó arbitrário interno
    
    casos = [
        {'TR': 10, 'TC': 20},
        {'TR': 50, 'TC': 20},
        {'TR': 10, 'TC': 80}
    ]
    T_obs = []
    
    for c in casos:
        placa = PlacaTermica(Lx, Ly, Nx, Ny, k=k_nominal, fonte_calor=fonte_calor_nominal)

        borda = borda_constante(Nx, Ny, c['TR'])
        circulo = circulo_constante(
            T  = c['TC'],
            r  = raio_circulo,
            cx = x_circulo,
            cy = y_circulo,
            Nx = placa.Nx,
            Ny = placa.Ny,
            hx = placa.hx,
            hy = placa.hy
        )

        fronteira_total = borda + circulo
        
        placa.resolver(fronteira_total)
        T_obs.append(placa.T[ic_escolhido])
        
    # Monta sistema linear para encontrar a, b e c:
    # [TR1, TC1, 1] [a] = [Tk1]
    # [TR2, TC2, 1] [b] = [Tk2]
    # [TR3, TC3, 1] [c] = [Tk3]
    Matriz_coefs = np.array([
        [casos[0]['TR'], casos[0]['TC'], 1],
        [casos[1]['TR'], casos[1]['TC'], 1],
        [casos[2]['TR'], casos[2]['TC'], 1]
    ])
    Vetor_Tks = np.array(T_obs)
    
    a, b, c_coef = np.linalg.solve(Matriz_coefs, Vetor_Tks)
    print(f"Nó k={ic_escolhido}")
    print(f"Coeficientes calculados: a = {a:.4f}, b = {b:.4f}, c = {c_coef:.4f}")
    print(f"Equação: Tk = {a:.4f}*TR + {b:.4f}*TC + {c_coef:.4f}")

def exercicio_3_extra(T_estrela:float=39.5):
    print("\n--- EXERCÍCIO 3 EXTRA ---")
    placa = PlacaTermica(Nx=161, Ny=81, k=k_nominal, Lx=Lx, Ly=Ly, fonte_calor=fonte_calor_nominal)
    fronteira = borda_padrao(placa.Nx, placa.Ny)

    Tc_ideal = placa.descobrir_Tc_para_Tmax(
        T_alvo=T_estrela, 
        fronteira=fronteira,
        raio=raio_circulo, cx=x_circulo, cy=y_circulo
    )

    return Tc_ideal

def exercicio_2_extra():
    print("\n--- EXERCÍCIO 2 EXTRA ---")
    placa = PlacaTermica(Nx=41, Ny=21, k=k_nominal, Lx=Lx, Ly=Ly, fonte_calor=fonte_calor_nominal)
    borda = borda_padrao(placa.Nx, placa.Ny)

    historico_jac = placa.gerar_historico_jacobi(borda, max_iter=1000, frame_skip=5)
    historico_gs = placa.gerar_historico_gauss_seidel(borda, max_iter=1000, omega=1.0, frame_skip=5)
    placa.animar_comparacao(historico_jac, historico_gs, intervalo_ms=40)
    
    placa.resolver(borda)

    placa.plota_placa(flag_type='contour')
    placa.plota_placa(flag_type='surface')

def exercicio_1_extra():
    print("\n--- EXERCÍCIO 1 EXTRA ---")
    ex_1_extra_tolerancia()
    ex_1_extra_malha()

def ex_1_extra_tolerancia():
    print("\n--- GRÁFICO TEMPO X TOLERÂNCIA ---")
    Nx, Ny = 101, 51

    placa = PlacaTermica(
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
        k=k_nominal,
        fonte_calor=fonte_calor_nominal
    )

    borda = borda_padrao(Nx, Ny)

    tolerancias = list(map(lambda n:10**(-n), range(2,14)))

    tempo_jacobi = []
    tempo_gs = []

    print("\nTolerância | Jacobi (s)   | Gauss-Seidel (s)")

    for tol in tolerancias:
        placa.resolver_jacobi(borda, tol, max_iter=1000000)
        t1 = placa.tempos_execucao['total']

        placa.resolver_gauss_seidel_rb(borda, tol, omega=1.85, max_iter=1000)
        t2 = placa.tempos_execucao['total']

        tempo_jacobi.append(t1)
        tempo_gs.append(t2)

        print(f"{tol:.2e}".ljust(10), f"| {t1:.6e} | {t2:.6e}")

    plt.figure(figsize=(6,4))
    plt.plot(tolerancias, tempo_jacobi, 'r-o', label='Jacobi')
    plt.plot(tolerancias, tempo_gs, 'b-o', label='Gauss-Seidel (SOR)')

    plt.xlabel('Tolerância (°C)')
    plt.ylabel('Tempo (s)')

    plt.xscale('log')
    # plt.yscale('log')

    plt.gca().invert_xaxis()

    plt.title(f'Tempo de execução por tolerância ({Nx}x{Ny})')
    plt.grid()
    plt.legend()
    plt.show()

def ex_1_extra_malha():
    print("\n--- GRÁFICO TEMPO X SUBDIVISÕES ---")
    
    subdivisoes = list(map(lambda n: (n*10+1,n*5+1), range(2, 33)))
    tol = 1e-12

    num_celulas = []
    tempo_jacobi = []
    tempo_gauss_seidel = []

    print("\nSubdivisões | Jacobi (s)   | Gauss-Seidel (s)")

    for (Nx, Ny) in subdivisoes:
        N_total = Nx * Ny

        placa = PlacaTermica(
            Lx=Lx,
            Ly=Ly,
            Nx=Nx,
            Ny=Ny,
            k=k_nominal,
            fonte_calor=fonte_calor_nominal
        )

        borda = borda_padrao(Nx, Ny)

        placa.resolver_jacobi(borda, tol, max_iter=1000000)
        t1 = placa.tempos_execucao['total']

        placa.resolver_gauss_seidel_rb(borda, tol, omega=1.85, max_iter=10000)
        t2 = placa.tempos_execucao['total']

        num_celulas.append(N_total)
        tempo_jacobi.append(t1)
        tempo_gauss_seidel.append(t2)

        print(f"{N_total}".ljust(11), f"| {t1:.6e} | {t2:.6e}")

    plt.figure(figsize=(6,4))
    plt.plot(num_celulas, tempo_jacobi, 'r-o', label='Jacobi')
    plt.plot(num_celulas, tempo_gauss_seidel, 'b-o', label='Gauss-Seidel (SOR)')

    plt.xlabel('Número de subdivisões')
    plt.ylabel('Tempo (s)')

    # plt.xscale('log')
    # plt.yscale('log')

    plt.title(f'Tempo de execução por subdivisões (Tolerância={tol})')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # exercicio_1()
    # exercicio_2(T_c=30.0)
    # exercicio_3()
    # exercicio_4()
    # exercicio_5()

    exercicio_1_extra()
    # exercicio_2_extra()
    # exercicio_3_extra(T_estrela=30.0)