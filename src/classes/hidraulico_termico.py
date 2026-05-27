import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from src.classes.rede_hidraulica import RedeHidraulica
from src.classes.placa_termica import PlacaTermica

class HidraulicoTermico:
    def __init__(self, Nx, Ny):
        self.placa = PlacaTermica(
            Lx=0.03,
            Ly=0.015,
            Nx=Nx,
            Ny=Ny,
            k=0.25,
            R=0.0025,
            fonte_calor=5e5
        )
        self.rede = RedeHidraulica(levels=3)

    # =======================================================================
    # MÉTODOS ORIGINAIS MANTIDOS (Viscosidade, Integração, etc.)
    # =======================================================================
    
    def calcular_viscosidade(self, T):
        return 0.001791 / (1.0 + 0.03368 * T + 0.000221 * (T**2))

    def distancia_ponto_segmento(self, p, a, b):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    # ... [MANTENHA AQUI SEUS MÉTODOS DE INTEGRAÇÃO DE LINHA E PLOTAGEM] ...

    # =======================================================================
    # NOVA ARQUITETURA: VOLUMES FINITOS PARA O EXERCÍCIO 1
    # =======================================================================

    def calcular_k_faces(self, dmax):
        """
        Calcula a condutividade modificada exatamente nas FACES entre os nós,
        garantindo a conservação correta do fluxo de calor.
        """
        Nx, Ny = self.placa.Nx, self.placa.Ny
        dx = self.placa.Lx / (Nx - 1)
        dy = self.placa.Ly / (Ny - 1)
        k_base = self.placa.k

        # kx_faces: avaliado no ponto médio entre (i,j) e (i+1,j)
        kx_faces = np.full((Nx - 1, Ny), k_base)
        # ky_faces: avaliado no ponto médio entre (i,j) e (i,j+1)
        ky_faces = np.full((Nx, Ny - 1), k_base)

        # 1. Varredura nas faces horizontais (X)
        for i in range(Nx - 1):
            for j in range(Ny):
                px = (i + 0.5) * dx # Ponto médio em x
                py = j * dy         # Alinhado com o nó em y
                p = np.array([px, py])
                
                soma = 0.0
                for n1, n2 in self.rede.conectividade:
                    a = self.rede.posicoes_nos[n1]
                    b = self.rede.posicoes_nos[n2]
                    d = self.distancia_ponto_segmento(p, a, b)
                    if d < dmax:
                        # CORREÇÃO: Fórmula com normalização (d / dmax)
                        soma += 1.0 / (1.0 + (d / dmax))
                kx_faces[i, j] = k_base * (1.0 + soma)

        # 2. Varredura nas faces verticais (Y)
        for i in range(Nx):
            for j in range(Ny - 1):
                px = i * dx         # Alinhado com o nó em x
                py = (j + 0.5) * dy # Ponto médio em y
                p = np.array([px, py])
                
                soma = 0.0
                for n1, n2 in self.rede.conectividade:
                    a = self.rede.posicoes_nos[n1]
                    b = self.rede.posicoes_nos[n2]
                    d = self.distancia_ponto_segmento(p, a, b)
                    if d < dmax:
                        soma += 1.0 / (1.0 + (d / dmax))
                ky_faces[i, j] = k_base * (1.0 + soma)

        return kx_faces, ky_faces

    def resolver_sistema_ex1(self, kx_faces, ky_faces, Tc=35.0):
        """
        Monta e resolve o sistema linear esparso utilizando as condutâncias 
        das faces calculadas previamente.
        """
        Nx, Ny = self.placa.Nx, self.placa.Ny
        dx = self.placa.Lx / (Nx - 1)
        dy = self.placa.Ly / (Ny - 1)
        nunk = Nx * Ny

        rows, cols, data = [], [], []
        b = np.zeros(nunk)
        
        # Centro do círculo (assumindo centro da placa)
        xc, yc = self.placa.Lx / 2.0, self.placa.Ly / 2.0
        R = self.placa.R

        for j in range(Ny):
            for i in range(Nx):
                # Usando seu indexador global original
                Ic = i + j * Nx 
                
                x_coord = i * dx
                y_coord = j * dy
                dist_centro = np.sqrt((x_coord - xc)**2 + (y_coord - yc)**2)

                # Condição 1: Nó dentro da inclusão circular (Temperatura Fixa)
                if dist_centro <= R:
                    rows.append(Ic); cols.append(Ic); data.append(1.0)
                    b[Ic] = Tc

                # Condição 2: Condições de contorno das bordas
                # ATENÇÃO: Ajuste as temperaturas das bordas (TR, TL, TT, TB) de 
                # acordo com o que a sua PlacaTermica original define!
                elif i == Nx - 1: # Borda Direita
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = 20.0 # Exemplo
                elif i == 0:      # Borda Esquerda
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = 20.0 # Exemplo
                elif j == Ny - 1: # Borda Superior
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = 20.0 # Exemplo
                elif j == 0:      # Borda Inferior
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = 20.0 # Exemplo
                
                # Equação de Balanço de Energia (Nó interno)
                else:
                    Ie = (i + 1) + j * Nx
                    Iw = (i - 1) + j * Nx
                    In = i + (j + 1) * Nx
                    Is = i + (j - 1) * Nx

                    ke = kx_faces[i, j]
                    kw = kx_faces[i - 1, j]
                    kn = ky_faces[i, j]
                    ks = ky_faces[i, j - 1]

                    ce = ke * dy / dx
                    cw = kw * dy / dx
                    cn = kn * dx / dy
                    cs = ks * dx / dy

                    c_central = ce + cw + cn + cs

                    rows.append(Ic); cols.append(Ic); data.append(c_central)
                    rows.append(Ic); cols.append(Ie); data.append(-ce)
                    rows.append(Ic); cols.append(Iw); data.append(-cw)
                    rows.append(Ic); cols.append(In); data.append(-cn)
                    rows.append(Ic); cols.append(Is); data.append(-cs)

                    b[Ic] = self.placa.fonte_calor * dx * dy

        A_sparse = sparse.coo_matrix((data, (rows, cols)), shape=(nunk, nunk)).tocsr()
        T_resolvido = spsolve(A_sparse, b)
        return T_resolvido

    def exercicio_1_2(self):
        dmax_list = [0.00025, 0.0005, 0.001]
        malhas = [(61, 31), (121, 61), (241, 121)]
        resultados = []

        for Nx, Ny in malhas:
            print(f"\n====================\nMALHA: {Nx} x {Ny}\n====================")
            
            for dmax in dmax_list:
                print(f"\n--- dmax = {dmax} ---")
                inicio = time.perf_counter()
                
                # Instanciamos o sistema fresco para evitar reuso sujo de cache
                sistema = HidraulicoTermico(Nx, Ny)
                dx = sistema.placa.Lx / (Nx - 1)
                dy = sistema.placa.Ly / (Ny - 1)

                # 1. Calcula os perfis de K nas interfaces (FACES)
                kx_faces, ky_faces = sistema.calcular_k_faces(dmax)

                # 2. Resolve a física utilizando a formulação por Volumes Finitos
                T_array = sistema.resolver_sistema_ex1(kx_faces, ky_faces, Tc=35.0)
                
                # Atualiza a placa com o resultado
                sistema.placa.T = T_array
                T_grid = T_array.reshape((Ny, Nx)).T # Transposto para bater com o meshgrid
                
                tempo_total = time.perf_counter() - inicio

                # 3. Plotagem
                X_plot, Y_plot = np.meshgrid(np.linspace(0, sistema.placa.Lx, Nx), 
                                             np.linspace(0, sistema.placa.Ly, Ny), indexing='ij')
                
                plt.figure(figsize=(6, 4))
                plt.contourf(X_plot, Y_plot, T_grid, 50, cmap='jet')
                plt.colorbar(label="Temperatura (°C)")
                plt.title(f"Temperatura - {Nx}x{Ny} - dmax={dmax}")
                plt.show()

                # Perfis 1D
                mid_vertical = T_grid[:, Ny // 2]
                mid_horizontal = T_grid[Nx // 2, :]

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.plot(mid_vertical)
                ax1.set_title("Perfil vertical (centro)")
                ax1.set_xlabel("y")
                ax1.set_ylabel("T")

                ax2.plot(mid_horizontal)
                ax2.set_title("Perfil horizontal (centro)")
                ax2.set_xlabel("x")
                
                plt.tight_layout()
                plt.show()

                Tmax = np.max(T_grid)

                # Armazena os dados do DataFrame
                resultados.append({
                    "malha": f"{Nx}x{Ny}",
                    "dmax": dmax,
                    "Tmax": Tmax,
                    "tempo_total": tempo_total
                })

        return pd.DataFrame(resultados)
    

def ex_2_acoplamento():
    acoplamento = HidraulicoTermico(241, 121)
    for method in ['linear', 'nearest', 'cubic']:
        acoplamento.mapa_contorno_grade_secundaria(101, 51, method=method)
    
    acoplamento_reduzido = HidraulicoTermico(61, 31)
    for method in ['linear', 'nearest', 'cubic']:
        acoplamento_reduzido.mapa_contorno_grade_secundaria(41, 21, method=method)
    
    acoplamento.plotar_rede_termica(method='linear')


def ex_3_acoplamento():
    acoplamento = HidraulicoTermico(241, 121)
    configs = [
        ('monte_carlo', 10), ('monte_carlo', 100),
        ('ponto_medio', 10), ('ponto_medio', 100),
        ('trapezio', 1), ('trapezio', 10), ('trapezio', 100)
    ]

    for metodo, n in configs:
        Tmed, tempo = acoplamento.temperaturas_medias_arestas(metodo=metodo, n_sub=n)
        print(f'Método: {metodo} | Subdivisões: {n} | Tempo: {tempo:.6f} s | TMédia: {Tmed.mean():.6f}')

    Tmed_plot, _ = acoplamento.temperaturas_medias_arestas(metodo='trapezio', n_sub=100)
    acoplamento.plotar_dados_arestas(Tmed_plot, label='Temperatura Média (°C)')


def ex_4_acoplamento():
    print("\n--- EXERCÍCIO 4: ANÁLISE DE CONVERGÊNCIA ---")
    malhas = [(61, 31), (121, 61), (241, 121)]
    configuracoes = [('ponto_medio', 10), ('trapezio', 10), ('trapezio', 100)]

    resultados = []

    for Nx, Ny in malhas:
        print(f"Processando malha: {Nx}x{Ny}...")
        sistema = HidraulicoTermico(Nx, Ny)
        
        # Warm-up: executa uma vez sem medir para forçar alocações e aquecer o cache
        sistema.atualizar_condutancias_ex4(metodo='trapezio', n_sub=10)
        
        for metodo, n_sub in configuracoes:
            # Medição com média de 5 execuções para eliminar ruído do OS
            tempos = []
            for _ in range(5):
                inicio = time.perf_counter()
                sistema.atualizar_condutancias_ex4(metodo=metodo, n_sub=n_sub)
                tempos.append(time.perf_counter() - inicio)

            resultados.append({

                'Malha': f'{Nx}x{Ny}',
                'Método': metodo,
                'Subdivisões': n_sub,
                'P_max (Pa)': sistema.rede.pressao.max(),
                'Potência (W)': sistema.rede.calcular_potencia(),
                'Tempo_Medio (s)': np.mean(tempos)
            })

    # Formatação limpa do DataFrame
    df = pd.DataFrame(resultados)
    #pd.options.display.float_format = '{:.6e}'.format
    print("\n" + df.to_string(index=False))

    # Visualização final focada na malha mais refinada
    print("\nGerando visualização da convergência...")
    sistema.plotar_dados_arestas(sistema.temperaturas_medias_arestas(metodo='trapezio', n_sub=100)[0], label='Temperatura Média (°C)')
    sistema.plotar_rede_termica(method='linear')



def calcular_termo_fonte_gaussiano(Nx, Ny, Lx, Ly, coord, edges, S0, distribuicao, d_max=0.001):
    sigma = d_max / 2.0
    x_grid = np.linspace(0.0, Lx, Nx)
    y_grid = np.linspace(0.0, Ly, Ny)
    Xs, Ys = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    P = np.column_stack([Xs.ravel(), Ys.ravel()])
    N_pts = P.shape[0]
    
    num_arestas = len(edges)
    I = np.zeros(num_arestas)
    
    if distribuicao == 'homogenea':
        I[:] = 1.0
    elif distribuicao == 'espinha':
        I[:] = 0.1
        y_centro = 0.5 * Ly
        tol = 1e-6
        for j, (idx_a, idx_b) in enumerate(edges):
            if abs(coord[idx_a, 1] - y_centro) < tol and abs(coord[idx_b, 1] - y_centro) < tol:
                I[j] = 100.0
                
    soma_gaussiana = np.zeros(N_pts)
    
    for k, (idx_a, idx_b) in enumerate(edges):
        a = coord[idx_a]
        b = coord[idx_b]
        ab = b - a
        ap = P - a
        ab_len_sq = np.sum(ab**2)
        
        if ab_len_sq > 0:
            t = np.dot(ap, ab) / ab_len_sq
            t = np.clip(t, 0.0, 1.0)
            p_proj = a + t[:, np.newaxis] * ab
            d = np.linalg.norm(P - p_proj, axis=1)
            
            dentro_do_raio = d <= d_max
            soma_gaussiana[dentro_do_raio] += I[k] * np.exp(-(d[dentro_do_raio]**2) / (2.0 * (sigma**2)))
            
    Sp = (S0 * soma_gaussiana).reshape(Nx, Ny)
    return Sp


def ex_5_acoplamento():
    print("\n--- EXERCÍCIO 5: COMPARAÇÃO DE MODELAGEM DA VISCOSIDADE ---")
    sistema = HidraulicoTermico(241, 121) 
    
    sistema.atualizar_condutancias_ex4(metodo='trapezio', n_sub=100)
    P4 = sistema.rede.pressao.max()
    Pot4 = sistema.rede.calcular_potencia()
    
    viscosidades_efetivas = sistema.atualizar_condutancias_ex5(metodo='trapezio', n_sub=100)
    P5 = sistema.rede.pressao.max()
    Pot5 = sistema.rede.calcular_potencia()
    
    diff_P = abs(P5 - P4) / P4 * 100
    diff_Pot = abs(Pot5 - Pot4) / Pot4 * 100
    
    print(f"Método Ex 4 [ mu(<T>) ]: P_max = {P4:.6e} Pa | Potência = {Pot4:.6e} W")
    print(f"Método Ex 5 [ <mu(T)> ]: P_max = {P5:.6e} Pa | Potência = {Pot5:.6e} W")
    print("-" * 50)
    print(f"Erro Relativo na Pressão Máxima: {diff_P:.4f}%")
    print(f"Erro Relativo na Potência:       {diff_Pot:.4f}%")

    sistema.plotar_dados_arestas(viscosidades_efetivas, label='Viscosidade Efetiva')
    sistema.plotar_rede_termica(method='linear')

def ex_2_extra():
    print("\n" + "="*50)
    print("   EXECUTANDO EXERCÍCIO 7: TERMO FONTE GAUSSIANO  ")
    print("="*50)
    
    Nx, Ny = 241, 121  
    Lx, Ly = 0.03, 0.015
    d_max = 0.001  
    
    sim_base = HidraulicoTermico(Nx, Ny)
    coord = sim_base.rede.posicoes_nos
    edges = sim_base.rede.conectividade
    
    S0_valores = [1e5, -1e5, 5e5, -5e5, 1e6, -1e6]
    distribuicoes = ['homogenea', 'espinha']
    temperaturas_maximas = {}
    
    for dist in distribuicoes:
        temperaturas_maximas[dist] = []
        perfis_horizontais = []
        perfis_verticais = []
        
        print(f"\n>> Iniciando análise para Distribuição: {dist.upper()}")
        
        for S0 in S0_valores:
            print(f"   Calculando cenário S0 = {S0:+.1e} ...")
            
            # 1. Matriz Gaussiana Sp
            Sp = calcular_termo_fonte_gaussiano(Nx, Ny, Lx, Ly, coord, edges, S0, dist, d_max)
            
            # 2. Fonte total combinada
            fonte_total = 5e5 + Sp
            
            # 3. Nova placa térmica com termo fonte espacialmente correto
            placa_modificada = PlacaTermica(
                Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, k=0.25, R=0.0025,
                fonte_calor=fonte_total.T  
            )
            placa_modificada.resolver_circulo(Tc=35, mode='sparse')
            
            # 4. Extração de resultados
            interpolador = placa_modificada.criar_interpolador('linear')
            
            x_g = np.linspace(0.0, Lx, Nx)
            y_g = np.linspace(0.0, Ly, Ny)
            Xs, Ys = np.meshgrid(x_g, y_g, indexing='ij')
            pts_all = np.column_stack([Xs.ravel(), Ys.ravel()])
            T_all = interpolador(pts_all)
            
            T_max = np.max(T_all)
            temperaturas_maximas[dist].append((S0, T_max))
            
            # --- MAPA DE CONTORNO BI-DIMENSIONAL ---
            fig, ax = plt.subplots(figsize=(8, 3.8))
            T_grid = T_all.reshape(Nx, Ny)
            cont = ax.contourf(Xs, Ys, T_grid, 20, cmap='jet')
            ax.contour(Xs, Ys, T_grid, 20, colors='k', linewidths=0.2)
            ax.set_aspect('equal')
            ax.set_title(f'Mapa de Temperatura: {dist.capitalize()} (S0 = {S0:+.1e})')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            plt.colorbar(cont, ax=ax, label='Temperatura (°C)')
            plt.tight_layout()
            plt.show()
            
            # --- EXTRAÇÃO DOS PERFIS UNIDIMENSIONAIS ---
            x_perf = np.linspace(0.0, Lx, 300)
            y_perf = np.ones_like(x_perf) * (Ly / 2.0)
            T_horiz = interpolador(np.column_stack([x_perf, y_perf]))
            perfis_horizontais.append((S0, T_horiz))
            
            y_perf_v = np.linspace(0.0, Ly, 300)
            x_perf_v = np.ones_like(y_perf_v) * (Lx / 2.0)
            T_vert = interpolador(np.column_stack([x_perf_v, y_perf_v]))
            perfis_verticais.append((S0, T_vert))
            
        # --- PLOT COMPARATIVO DOS PERFIS DA DISTRIBUIÇÃO ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        for S0, T_h in perfis_horizontais:
            ax1.plot(x_perf, T_h, label=f'S0 = {S0:+.1e}')
        ax1.set_title(f'Perfis Horizontais (y = Ly/2) - {dist.capitalize()}')
        ax1.set_xlabel('Posição X (m)')
        ax1.set_ylabel('Temperatura (°C)')
        ax1.legend()
        ax1.grid(True)
        
        for S0, T_v in perfis_verticais:
            ax2.plot(T_v, y_perf_v, label=f'S0 = {S0:+.1e}')
        ax2.set_title(f'Perfis Verticais (x = Lx/2) - {dist.capitalize()}')
        ax2.set_xlabel('Temperatura (°C)')
        ax2.set_ylabel('Posição Y (m)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    print("\n" + "="*60)
    print("      TABELA COMPARATIVA DE TEMPERATURAS MÁXIMAS OBTIDAS")
    print("="*60)
    for dist in distribuicoes:
        print(f"\nDistribuição: {dist.upper()}")
        print(f"{'Intensidade S0':<15} | {'Temperatura Máxima (°C)':<25}")
        print("-"*45)
        for S0, T_max in temperaturas_maximas[dist]:
            print(f"{S0:+.1e}        | {T_max:.4f} °C")
    print("="*60)


########################################################################
# DISTÂNCIA ENTRE PONTO E SEGMENTO
########################################################################



def ex_1_especial_acoplamento():
    print("\n--- INICIANDO ROTINA ESPECIAL: MAPA DE PROXIMIDADE E CONDUTIVIDADE ---")
    
    # Instanciamos uma malha de resolução baixa apenas como gatilho para acessar o método,
    # já que o próprio exercicio_1_2 fará a instanciação das malhas de teste internamente.
    sistema_gatilho = HidraulicoTermico(61, 31)
    
    print("Iniciando varredura de malhas e cálculo de distâncias (isso pode demorar devido aos laços não-vetorizados)...")
    
    # Executa a rotina que gera os gráficos de contorno e perfis 1D
    df_resultados = sistema_gatilho.exercicio_1_2()
    
    print("\n=======================================================")
    print("RESULTADOS GLOBAIS: TEMPERATURA MÁXIMA vs DISTÂNCIA (dmax)")
    print("=======================================================")
    print(df_resultados.to_string(index=False))
    
