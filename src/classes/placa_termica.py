import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

class PlacaTermica:
    def __init__(self, Lx: float, Ly: float, Nx: int, Ny: int, 
                 condutividade=None, fonte_calor: float = 0.0):
        """Construtor da placa térmica."""
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.N_total = Nx * Ny
        
        # Espaçamentos
        self.hx = Lx / (Nx - 1)
        self.hy = Ly / (Ny - 1)
        
        # Condutividade pode ser um float (constante) ou uma função k(x,y)
        self.k = condutividade if condutividade is not None else 1.0
        self.fonte_calor = fonte_calor
        
        self.circ_region = None # (xc, yc, R, Tc)
        
        self.temperaturas = None
        self.tempos_execucao = {} # Guarda o tempo de montagem e resolução

    def set_regiao_circular(self, xc, yc, R, Tc):
        """Define a condição de contorno circular interna."""
        self.circ_region = (xc, yc, R, Tc)

    def flatten_coordinate(self, i: int, j: int) -> int:
        return i + j * self.Nx

    def _get_k(self, x, y):
        """Avalia a condutividade no ponto (x,y) se for função, senão retorna o valor constante."""
        if callable(self.k):
            return self.k(x, y)
        return self.k

    def _get_boundary_value(self, boundary, x, y):
        """Retorna o valor de contorno, aceitando escalar ou função."""
        if callable(boundary):
            return boundary(x, y)
        return boundary

    def resolver(self, T_top: float, T_bottom: float, T_left: float, T_right: float, use_sparse=True):
        """Monta e resolve o sistema. Retorna as temperaturas."""
        t_inicio_montagem = time.time()
        
        # Listas para construção da matriz esparsa (formato COO é o mais rápido para montar)
        rows, cols, data = [], [], []
        b = np.zeros(self.N_total)
        
        # Se formos forçar matriz densa (apenas para comparar tempos em malhas pequenas)
        A_densa = np.zeros((self.N_total, self.N_total)) if not use_sparse else None

        for j in range(self.Ny):
            for i in range(self.Nx):
                ic = self.flatten_coordinate(i, j)
                x = i * self.hx
                y = j * self.hy

                T_top_val = self._get_boundary_value(T_top, x, y)
                T_bottom_val = self._get_boundary_value(T_bottom, x, y)
                T_left_val = self._get_boundary_value(T_left, x, y)
                T_right_val = self._get_boundary_value(T_right, x, y)
                
                # 1. Checa Condições de Contorno nas Bordas
                if i == 0:
                    val, contorno = 1.0, T_left_val
                elif i == self.Nx - 1:
                    val, contorno = 1.0, T_right_val
                elif j == 0:
                    val, contorno = 1.0, T_bottom_val
                elif j == self.Ny - 1:
                    val, contorno = 1.0, T_top_val
                else:
                    # 2. Checa Condição de Contorno Interna (Região Circular)
                    if self.circ_region:
                        xc, yc, R, Tc = self.circ_region
                        if (x - xc)**2 + (y - yc)**2 <= R**2:
                            rows.append(ic); cols.append(ic); data.append(1.0)
                            b[ic] = Tc
                            if not use_sparse: A_densa[ic, ic] = 1.0
                            continue
                    
                    # 3. Nó interno padrão (Equação de Balanço)
                    # Avalia condutividades nas faces
                    ke = self._get_k(x + self.hx/2, y)
                    kw = self._get_k(x - self.hx/2, y)
                    kn = self._get_k(x, y + self.hy/2)
                    ks = self._get_k(x, y - self.hy/2)
                    
                    ie = self.flatten_coordinate(i + 1, j)
                    iw = self.flatten_coordinate(i - 1, j)
                    inorth = self.flatten_coordinate(i, j + 1)
                    isouth = self.flatten_coordinate(i, j - 1)
                    
                    coef_central = ke + kw + kn + ks
                    
                    rows.extend([ic, ic, ic, ic, ic])
                    cols.extend([ic, ie, iw, inorth, isouth])
                    data.extend([coef_central, -ke, -kw, -kn, -ks])
                    
                    b[ic] = self.fonte_calor * (self.hx * self.hy)
                    
                    if not use_sparse:
                        A_densa[ic, ic] = coef_central
                        A_densa[ic, ie] = -ke
                        A_densa[ic, iw] = -kw
                        A_densa[ic, inorth] = -kn
                        A_densa[ic, isouth] = -ks
                    continue
                
                # Se caiu numa condição de borda, aplica aqui
                rows.append(ic); cols.append(ic); data.append(val)
                b[ic] = contorno
                if not use_sparse: A_densa[ic, ic] = val

        self.tempos_execucao['montagem'] = time.time() - t_inicio_montagem
        
        t_inicio_solucao = time.time()
        if use_sparse:
            # Cria a matriz esparsa e resolve
            A_esparsa = sp.csr_matrix((data, (rows, cols)), shape=(self.N_total, self.N_total))
            self.temperaturas = spla.spsolve(A_esparsa, b)
        else:
            # Resolve pelo método denso padrão
            self.temperaturas = np.linalg.solve(A_densa, b)
            
        self.tempos_execucao['resolucao'] = time.time() - t_inicio_solucao
        self.tempos_execucao['total'] = self.tempos_execucao['montagem'] + self.tempos_execucao['resolucao']
        
        return self.temperaturas

    def get_max_temp(self):
        return np.max(self.temperaturas)

    def get_avg_temp(self):
        return np.mean(self.temperaturas)

    def get_central_profile(self):
        """Retorna o perfil de temperaturas ao longo do eixo X central (y = Ly/2)."""
        j_center = self.Ny // 2
        Z = np.copy(self.temperaturas).reshape(self.Ny, self.Nx)
        x = np.linspace(0.0, self.Lx, self.Nx)
        return x, Z[j_center, :]

    def plotaPlaca(self, flag_type='contour', filename=None, title='Distribuição de Temperatura'):
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
        

# --- PARÂMETROS BASE SUGERIDOS ---
Lx, Ly = 0.02, 0.01  # 2 cm x 1 cm em metros
k_nominal = 0.25
fonte_calor_nominal = 5e5
raio_regiao_circular = 0.002
T_bordas = {'T_top': 10, 'T_bottom': 10, 'T_left': 0, 'T_right': 20}

def exercicio_1():
    print("\n--- EXERCÍCIO 1 ---")
    malhas = [(21, 11), (41, 21), (81, 41), (161, 81), (321, 161)]
    
    print(f"{'Malha':<12} | {'T_max':<8} | {'Tempo Esparsa (s)':<18} | {'Tempo Densa (s)':<15}")
    print("-" * 65)
    
    for Nx, Ny in malhas:
        placa = PlacaTermica(Lx, Ly, Nx, Ny, condutividade=k_nominal)
        
        # Teste com Matriz Esparsa
        placa.resolver(**T_bordas, use_sparse=True)
        t_esparsa = placa.tempos_execucao['total']
        t_max = placa.get_max_temp()
        
        # Teste com Matriz Densa (CUIDADO com falta de memória para malhas grandes)
        t_densa = "N/A (Memória)"
        if Nx * Ny <= 20000: # Proteção contra MemoryError
            placa.resolver(**T_bordas, use_sparse=False)
            t_densa = f"{placa.tempos_execucao['total']:.5f}"
        
        print(f"({Nx}, {Ny})".ljust(12) + f" | {t_max:.2f}   | {t_esparsa:.5f}".ljust(22) + f" | {t_densa}")
        
        # Plotar apenas para uma malha intermediária para não abrir 50 janelas
        if Nx == 81:
            placa.plotaPlaca(title=f"Ex 1: Contorno para malha {Nx}x{Ny}")
            
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

    def T_linear_em_x(x, y):
        return 10.0 + 20.0 * (x / Lx)

    for Nx, Ny in malhas:
        placa = PlacaTermica(
            Lx,
            Ly,
            Nx,
            Ny,
            condutividade=k_nominal,
            fonte_calor=fonte_calor_nominal,
        )
        
        # Inserindo a região circular interna (x_c, y_c, R, T_C)
        placa.set_regiao_circular(xc=0.75*Lx, yc=0.5*Ly, R=raio_regiao_circular, Tc=30.0)
        placa.resolver(
            T_top=T_linear_em_x,
            T_bottom=T_linear_em_x,
            T_left=10.0,
            T_right=30.0,
            use_sparse=True,
        )
        
        print(f"Malha {Nx}x{Ny} -> T_max = {placa.get_max_temp():.2f}")
        if Nx == 81:
            placa.plotaPlaca(title=f"Ex 2: Região Circular (Malha {Nx}x{Ny})")
            x_perfil, T_perfil = placa.get_central_profile()
            plt.figure()
            plt.plot(x_perfil, T_perfil, 'r-', label='Com Cilindro')
            plt.title("Ex 2: Perfil Central")
            plt.legend()
            plt.grid()
            plt.show()

def exercicio_3():
    print("\n--- EXERCÍCIO 3 (Condutividade Variável) ---")
    
    # Define a função k(x,y) do problema
    def k_variavel(x, y):
        return 0.2 + 0.05 * np.sin(3*np.pi*x / Lx) * np.sin(3*np.pi*y / Ly)
    
    placa = PlacaTermica(Lx, Ly, Nx=81, Ny=41, condutividade=k_variavel)
    placa.resolver(**T_bordas, use_sparse=True)
    
    print(f"Temperatura Máxima com k variável: {placa.get_max_temp():.2f}")
    placa.plotaPlaca(title="Ex 3: K Variável")

def exercicio_4():
    print("\n--- EXERCÍCIO 4 (Influência de T_C) ---")
    Nx, Ny = 101, 51
    temperaturas_Tc = np.linspace(0, 100, 10)
    maximas = []
    medias = []
    
    for Tc in temperaturas_Tc:
        placa = PlacaTermica(Lx, Ly, Nx, Ny, condutividade=k_nominal)
        placa.set_regiao_circular(xc=0.75*Lx, yc=0.5*Ly, R=raio_regiao_circular, Tc=Tc)
        placa.resolver(**T_bordas, use_sparse=True)
        maximas.append(placa.get_max_temp())
        medias.append(placa.get_avg_temp())
        
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
    k_escolhido = 233 # Índice do nó arbitrário interno
    
    casos = [
        {'TR': 10, 'TC': 20},
        {'TR': 50, 'TC': 20},
        {'TR': 10, 'TC': 80}
    ]
    T_obs = []
    
    for c in casos:
        placa = PlacaTermica(Lx, Ly, Nx, Ny, condutividade=k_nominal)
        placa.set_regiao_circular(xc=0.75*Lx, yc=0.5*Ly, R=raio_regiao_circular, Tc=c['TC'])
        # Aplicando a mesma temperatura TR em todas as bordas
        placa.resolver(T_top=c['TR'], T_bottom=c['TR'], T_left=c['TR'], T_right=c['TR'], use_sparse=True)
        T_obs.append(placa.temperaturas[k_escolhido])
        
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
    print(f"Nó k={k_escolhido}")
    print(f"Coeficientes calculados: a = {a:.4f}, b = {b:.4f}, c = {c_coef:.4f}")
    print(f"Equação: Tk = {a:.4f}*TR + {b:.4f}*TC + {c_coef:.4f}")

if __name__ == "__main__":
    # Remova os comentários das funções abaixo de acordo com qual quer executar:
    # exercicio_1()
    exercicio_2()
    # exercicio_3()
    # exercicio_4()
    # exercicio_5()