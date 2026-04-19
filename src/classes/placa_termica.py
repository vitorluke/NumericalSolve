from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

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

        self.fonte_calor = fonte_calor # Termo f(x,y)

        self.A = None
        self.b = None        
        self.temperaturas = None

        self.tempos_execucao = {}

    def flatten_coordinate(self, i: int, j: int) -> int:
        return i + j * self.Nx

    def montar_densa(self, fronteira):
        tempo_inicio = time.time()

        self.A = np.zeros((self.N_total, self.N_total))
        self.b = np.zeros(self.N_total)

        if callable(self.k):
            self.montar_densa_variavel()
        else:
            self.montar_densa_uniforme()

        for (ic, T) in fronteira:
            self.A[ic, :]  = 0.0
            self.A[ic, ic] = 1.0

            self.b[ic] = T

        tempo_fim = time.time()

        self.tempos_execucao['montagem'] = tempo_fim - tempo_inicio

    def montar_densa_uniforme(self):
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

    def montar_densa_variavel(self):        
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

        """
        # Impõe por padrão uma temperatura 0 nas bordas

        for i in range(self.Nx):
            i_top = self.flatten_coordinate(i, 0)
            i_bottom = self.flatten_coordinate(i, self.Ny - 1)

            self.A[i_top, i_top] = 1.0
            self.A[i_bottom, i_bottom] = 1.0

        for j in range(self.Ny):
            i_left = self.flatten_coordinate(0, j)
            i_right = self.flatten_coordinate(self.Nx - 1, j)

            self.A[i_left, i_left] = 1.0
            self.A[i_right, i_right] = 1.0

        """

    def resolver_densa(self, fronteira:list[(int, float)]):
        self.montar_densa(fronteira)

        tempo_inicio = time.time()

        self.temperaturas = np.linalg.solve(self.A, self.b)

        tempo_fim = time.time()

        self.tempos_execucao['resolucao'] = tempo_fim - tempo_inicio
        self.tempos_execucao['total'] = self.tempos_execucao['montagem'] + self.tempos_execucao['resolucao']

        return self.temperaturas
    
    def montar_esparsa(self, fronteira:list[(int, float)]):
        tempo_inicio = time.time()

        rows = []
        cols = []
        data = []

        self.b = np.zeros(self.N_total)

        for (ic, T) in fronteira:
            if ic in rows:
                continue
            
            rows.append(ic)
            cols.append(ic)
            data.append(1.0)

            self.b[ic] = T

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

        return self.temperaturas

    def resolver_esparsa(self, fronteira:list[(int, float)]):
        self.montar_esparsa(fronteira)

        """
        tempo_inicio = time.time()

        self.montar_densa(fronteira)
        self.A = sparse.csr_matrix(self.A)

        tempo_fim = time.time()

        self.tempos_execucao['montagem'] = tempo_fim - tempo_inicio

        """

        tempo_inicio = time.time()

        self.temperaturas = sparse.linalg.spsolve(self.A, self.b)

        tempo_fim = time.time()

        self.tempos_execucao['resolucao'] = tempo_fim - tempo_inicio
        self.tempos_execucao['total'] = self.tempos_execucao['montagem'] + self.tempos_execucao['resolucao']

        return self.temperaturas
    
    def resolver(self, fronteira:list[(int, float)], use_sparse:bool=True):
        if use_sparse:
            return self.resolver_esparsa(fronteira)
        else:
            return self.resolver_densa(fronteira)

    def temp_max(self):
        if self.temperaturas is None:
            return None
        else:
            return np.max(self.temperaturas)

    def temp_med(self):
        if self.temperaturas is None:
            return None
        else:
            return np.average(self.temperaturas)

    def plota_placa(self, flag_type='contour', title='Distribuição de temperatura', filename=None):
        """Plota a dispersão de calor da placa (contour 2D ou surface 3D)."""
        if self.temperaturas is None:
            print("Erro: Resolva a placa antes de plotar.")
            return

        x = np.linspace(0.0, self.Lx, self.Nx)
        y = np.linspace(0.0, self.Ly, self.Ny)
        X, Y = np.meshgrid(x, y)
        Z = np.copy(self.temperaturas).reshape(self.Ny, self.Nx)
        
        if flag_type == 'contour':
            fig, ax = plt.subplots(figsize=(8,4))
            ax.set_aspect('equal')
            ax.set(xlabel='x', ylabel='y', title=title)
            im = ax.contourf(X, Y, Z, 20, cmap='jet')
            im2 = ax.contour(X, Y, Z, 20, linewidths=0.25, colors='k')
            fig.colorbar(im, ax=ax, orientation='horizontal')
        elif flag_type == 'surface':
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8,6))
            surf = ax.plot_surface(X, Y, Z, cmap='jet')
            ax.set(xlabel='x', ylabel='y', zlabel='Temperatura', title=title)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5) 
            
        plt.xticks([0, self.Lx/2, self.Lx])
        plt.yticks([0, self.Ly/2, self.Ly])

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
        termo_fonte = (self.ds * self.fonte_calor) / self.k
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
        
        termo_fonte = (self.ds * self.fonte_calor) / self.k
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
        
        Lx = self.Lx
        Ly = self.Ly
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
                self.temperaturas = mapa_temp.flatten() # Salva para poder usar o plota_placa
                return Tc_novo
            
            # Atualiza os valores para o próximo loop
            Tc_0, erro_0 = Tc_1, erro_1
            Tc_1, erro_1 = Tc_novo, erro_novo

        print("Aviso: Limite de iterações atingido sem convergir perfeitamente.")
        return Tc_1
    
    def get_central_profile(self):
        """Retorna o perfil de temperaturas ao longo do eixo X central (y = Ly/2)."""
        j_center = self.Ny // 2
        Z = np.copy(self.temperaturas).reshape(self.Ny, self.Nx)
        x = np.linspace(0.0, self.Lx, self.Nx)
        return x, Z[j_center, :]

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
    malhas = [(21, 11), (41, 21), (81, 41), (161, 81), (321, 161)]
    
    print(f"{'Malha':<12} | {'T_max':<8} | {'Tempo Esparsa (s)':<18} | {'Tempo Densa (s)':<15}")
    print("-" * 65)
    
    for Nx, Ny in malhas:
        placa = PlacaTermica(Lx, Ly, Nx, Ny, k=k_nominal, fonte_calor=fonte_calor_nominal)
        temp_bordas = borda_padrao(Nx, Ny)
        
        # Teste com Matriz Esparsa
        placa.resolver(temp_bordas)
        tempo_esparsa = placa.tempos_execucao['total']
        t_max = placa.temp_max()
        
        # Teste com Matriz Densa (CUIDADO com falta de memória para malhas grandes)
        tempo_densa = "N/A (Memória)"
        if Nx * Ny <= 20000: # Proteção contra MemoryError
            placa.resolver(temp_bordas)
            tempo_densa = f"{placa.tempos_execucao['total']:.5f}"
        
        print(f"({Nx}, {Ny})".ljust(12) + f" | {t_max:.2f}   | {tempo_esparsa:.5f}".ljust(22) + f" | {tempo_densa}")
        
        # Plotar apenas para uma malha intermediária para não abrir 50 janelas
        if Nx == 321:
            placa.plota_placa(title=f"Ex 1: Contorno para malha {Nx}x{Ny}")
            
            # Plotar temperatura no eixo central
            x_perfil, T_perfil = placa.get_central_profile()
            plt.figure(figsize=(6,4))
            plt.plot(x_perfil, T_perfil, 'b-', label='y = Ly/2')
            plt.title("Ex 1: Perfil de Temperatura no Eixo Central")
            plt.xlabel("x")
            plt.ylabel("Temperatura")
            plt.grid(True)
            plt.legend()
            plt.show()

def exercicio_2():
    print("\n--- EXERCÍCIO 2 (Parâmetros do enunciado) ---")
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
            T  = 30.0,
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
            placa.plota_placa(title=f"Ex 2: Região Circular (Malha {Nx}x{Ny})")
            x_perfil, T_perfil = placa.get_central_profile()
            plt.figure()
            plt.plot(x_perfil, T_perfil, 'r-', label='Com Cilindro')
            plt.title("Ex 2: Perfil Central")
            plt.legend()
            plt.grid()
            plt.show()

def exercicio_3():
    print("\n--- EXERCÍCIO 3 (Condutividade Variável) ---")
    
    def k_variavel(x, y):
        return 0.2 + 0.05 * np.sin(3*np.pi*x / Lx) * np.sin(3*np.pi*y / Ly)

    Nx = 81
    Ny = 41

    placa = PlacaTermica(Lx, Ly, Nx, Ny, k=k_variavel, fonte_calor=fonte_calor_nominal)

    temp_bordas = borda_padrao(Nx, Ny)

    placa.resolver(temp_bordas)
    
    print(f"Temperatura Máxima com k variável: {placa.temp_max():.2f}")
    placa.plota_placa(title="Ex 3: K Variável")

def exercicio_4():
    print("\n--- EXERCÍCIO 4 (Influência de T_C) ---")
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
    plt.xlabel('Temperatura no Cilindro (T_C)')
    plt.ylabel('Temperatura na Placa')
    plt.title('Ex 4: T_max e T_media vs T_C')
    plt.grid()
    plt.legend()
    plt.show()

def exercicio_5():
    print("\n--- EXERCÍCIO 5 (Linearidade Tk = a*TR + b*TC + c) ---")
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
        T_obs.append(placa.temperaturas[ic_escolhido])
        
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

if __name__ == "__main__":
    exercicio_1()
    # exercicio_2()
    # exercicio_3()
    # exercicio_4()
    # exercicio_5()

    # placa = PlacaTermica(Nx=41, Ny=21, k=k_nominal, Lx=Lx, Ly=Ly, fonte_calor=fonte_calor_nominal)
    # fronteira = borda_padrao(placa.Nx, placa.Ny)

    # T_estrela = 39.5

    # Tc_ideal = placa.descobrir_Tc_para_Tmax(
    #     T_alvo=T_estrela, 
    #     fronteira=fronteira,
    #     raio=raio_circulo, cx=x_circulo, cy=y_circulo
    # )

    # placa.resolver(fronteira)

    # historico_jac = placa.gerar_historico_jacobi(fronteira, max_iter=200)
    # historico_gs = placa.gerar_historico_gauss_seidel(fronteira, max_iter=200)
    # placa.animar_comparacao(historico_jac, historico_gs, intervalo_ms=40)
    
    # placa.plota_placa(flag_type='contour')
    # placa.plota_placa(flag_type='surface')