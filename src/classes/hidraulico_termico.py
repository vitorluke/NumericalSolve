import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from src.classes.rede_hidraulica import RedeHidraulica
from src.classes.placa_termica import PlacaTermica

class HidraulicoTermico:
    def __init__(self, rede, placa):
        self.placa = placa
        self.rede = rede
        
        # =========================================================
        # CORREÇÃO DEFINITIVA: Escala e Distribuição da Rede
        # Garante que a rede ocupe apenas a região esquerda (0 a 0.02 m)
        # e fique perfeitamente centralizada no eixo Y.
        # =========================================================
        max_x = np.max(self.rede.posicoes_nos[:, 0])
        if max_x > 0:
            fator_escala = 0.02 / max_x
            self.rede.posicoes_nos[:, 0] *= fator_escala
            self.rede.posicoes_nos[:, 1] *= fator_escala  # Mantém a proporção geométrica
            
        y_min = np.min(self.rede.posicoes_nos[:, 1])
        y_max = np.max(self.rede.posicoes_nos[:, 1])
        y_mid = (y_min + y_max) / 2.0
        
        # Move a rede para o centro exato da placa (Ly / 2)
        self.rede.posicoes_nos[:, 1] += (self.placa.Ly / 2.0) - y_mid
        
        kx, ky = self.calcular_k_faces(dmax=0.001)
        self.placa.T = self.resolver_sistema_ex1(kx, ky, Tc=35.0)

    @classmethod
    def instantiate_subsystems(cls, Nx, Ny):
        placa = PlacaTermica(
            Lx=0.03,
            Ly=0.015,
            Nx=Nx,
            Ny=Ny,
            k=0.25,
            R=0.0025,
            fonte_calor=5e5
        )
        rede = RedeHidraulica(levels=3)

        return cls(placa, rede)

    # =======================================================================
    # MÉTODOS ORIGINAIS MANTIDOS (Viscosidade, Integração, etc.)
    # =======================================================================
    
    def calcular_viscosidade(self, T):
        return 0.001791 / (1.0 + 0.03368 * T + 0.000221 * (T**2))

    def distancia_ponto_segmento(self, p, a, b):
        ab = b - a
        ap = p - a
        ab_len_sq = np.dot(ab, ab)
        
        if ab_len_sq == 0.0:
            return np.linalg.norm(ap)
        
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0.0, 1.0)
        
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    def integrar_linha(self, p0, p1, func, metodo='trapezio', n_sub=100):
        if metodo == 'monte_carlo':
            ts = np.random.uniform(0.0, 1.0, n_sub)
            pts = (1.0 - ts[:, None]) * p0 + ts[:, None] * p1
            valores = func(pts)
            return np.mean(valores)
        
        if metodo == 'ponto_medio':
            ts = np.linspace(0.5 / n_sub, 1.0 - 0.5 / n_sub, n_sub)
            pts = (1.0 - ts[:, None]) * p0 + ts[:, None] * p1
            valores = func(pts)
            return np.mean(valores)

        if metodo == 'trapezio':
            ts = np.linspace(0.0, 1.0, n_sub + 1)
            pts = (1.0 - ts[:, None]) * p0 + ts[:, None] * p1
            valores = func(pts)
            return (np.sum(valores) - 0.5 * (valores[0] + valores[-1])) / n_sub

        raise ValueError()

    def temperatura_media_aresta(self, i, j, interpolador, metodo='trapezio', n_sub=100):
        p0 = self.rede.posicoes_nos[i]
        p1 = self.rede.posicoes_nos[j]
        
        def func_T(pts):
            return interpolador(pts).ravel()
            
        return self.integrar_linha(p0, p1, func_T, metodo, n_sub)

    def viscosidade_efetiva_aresta(self, i, j, interpolador, metodo='trapezio', n_sub=100):
        p0 = self.rede.posicoes_nos[i]
        p1 = self.rede.posicoes_nos[j]
        
        def func_mu(pts):
            T_vals = interpolador(pts).ravel()
            return self.calcular_viscosidade(T_vals)
            
        return self.integrar_linha(p0, p1, func_mu, metodo, n_sub)

    def temperaturas_medias_arestas(self, metodo='trapezio', n_sub=100, interp='linear'):
        interpolador = self.placa.criar_interpolador(interp)
        temperaturas = []
        inicio = time.perf_counter()

        for i, j in self.rede.conectividade:
            T_med = self.temperatura_media_aresta(i, j, interpolador, metodo, n_sub)
            temperaturas.append(T_med)

        fim = time.perf_counter()
        return np.array(temperaturas), fim - inicio

    def viscosidades_medias_arestas(self, metodo='trapezio', n_sub=100, interp='linear'):
        interpolador = self.placa.criar_interpolador(interp)
        viscosidades = []
        inicio = time.perf_counter()

        for i, j in self.rede.conectividade:
            mu_efetiva = self.viscosidade_efetiva_aresta(i, j, interpolador, metodo, n_sub)
            viscosidades.append(mu_efetiva)

        fim = time.perf_counter()
        return np.array(viscosidades), fim - inicio

    def plotar_dados_arestas(self, valores, label='Valor'):
        coord = self.rede.posicoes_nos
        edges = self.rede.conectividade
        fig, ax = plt.subplots(figsize=(10, 5))
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(valores.min(), valores.max())

        for k, (i, j) in enumerate(edges):
            x1, y1 = coord[i]
            x2, y2 = coord[j]
            cor = cmap(norm(valores[k]))
            ax.plot([x1, x2], [y1, y2], color=cor, linewidth=3)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label=label)
        ax.set_aspect('equal')
        plt.show()

    def temperaturas_nos(self, method='linear'):
        coords = self.rede.posicoes_nos
        interpolador = self.placa.criar_interpolador(method)
        return interpolador(coords)

    def plotar_rede_termica(self, method='linear'):
        temperaturas = self.temperaturas_nos(method)
        coord = self.rede.posicoes_nos
        edges = self.rede.conectividade
        fig, ax = plt.subplots(figsize=(10, 8))

        for (i, j) in edges:
            x1, y1 = coord[i]
            x2, y2 = coord[j]
            ax.plot([x1, x2], [y1, y2], color='black', linewidth=1.5, zorder=1)

        scatter = ax.scatter(
            coord[:, 0],
            coord[:, 1],
            c=temperaturas,
            cmap='jet',
            s=250,
            edgecolors='black',
            zorder=2
        )

        for idx, (x, y) in enumerate(coord):
            ax.text(x, y, f'{idx+1}', ha='center', va='center', fontweight='bold', color='white')

        plt.colorbar(scatter, ax=ax, label='Temperatura (°C)')
        ax.set_aspect('equal')
        ax.set_title(f'Temperatura nos nós ({method})')
        plt.show()

    def mapa_contorno_grade_secundaria(self, Nx_sec, Ny_sec, method='linear'):
        x_sec = np.linspace(0.0, self.placa.Lx, Nx_sec)
        y_sec = np.linspace(0.0, self.placa.Ly, Ny_sec)
        Xs, Ys = np.meshgrid(x_sec, y_sec, indexing='ij')
        pts = np.column_stack([Xs.ravel(), Ys.ravel()])
        interpolador = self.placa.criar_interpolador(method)
        T_sec = interpolador(pts).reshape(Nx_sec, Ny_sec)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        cont = ax.contourf(Xs, Ys, T_sec, 20, cmap='jet')
        ax.contour(Xs, Ys, T_sec, 20, colors='k', linewidths=0.3)
        ax.set_aspect('equal')
        ax.set_title(f'Interpolação {method} ({Nx_sec}x{Ny_sec})')
        plt.colorbar(cont, ax=ax)
        plt.show()

    def atualizar_condutancias_ex4(self, metodo='trapezio', n_sub=100):
        T_med, _ = self.temperaturas_medias_arestas(metodo=metodo, n_sub=n_sub)
        viscosidades = self.calcular_viscosidade(T_med)
        self.rede.atualizar_condutancias(viscosidades)
        self.rede.resolver()
        return T_med
        
    def atualizar_condutancias_ex5(self, metodo='trapezio', n_sub=100):
        viscosidades_efetivas, _ = self.viscosidades_medias_arestas(metodo=metodo, n_sub=n_sub)
        self.rede.atualizar_condutancias(viscosidades_efetivas)
        self.rede.resolver()
        return viscosidades_efetivas

    # =======================================================================
    # FÍSICA E PLOTAGEM CORRIGIDAS (EXERCÍCIO 1)
    # =======================================================================

    def calcular_k_faces(self, dmax):
        Nx, Ny = self.placa.Nx, self.placa.Ny
        dx = self.placa.Lx / (Nx - 1)
        dy = self.placa.Ly / (Ny - 1)
        k_base = self.placa.k

        kx_faces = np.full((Nx - 1, Ny), k_base)
        ky_faces = np.full((Nx, Ny - 1), k_base)

        edges = np.array(self.rede.conectividade)
        A = self.rede.posicoes_nos[edges[:, 0]]
        B = self.rede.posicoes_nos[edges[:, 1]]
        AB = B - A
        AB_len_sq = np.sum(AB**2, axis=1)
        
        valid = AB_len_sq > 0

        for i in range(Nx - 1):
            for j in range(Ny):
                px = (i + 0.5) * dx
                py = j * dy
                P = np.array([px, py])
                
                AP = P - A
                t = np.zeros(len(edges))
                t[valid] = np.sum(AP[valid] * AB[valid], axis=1) / AB_len_sq[valid]
                t = np.clip(t, 0.0, 1.0)
                proj = A + t[:, np.newaxis] * AB
                d = np.linalg.norm(P - proj, axis=1)
                
                mask = d < dmax
                kx_faces[i, j] = k_base * (1.0 + np.sum(1.0 / (1.0 + (d[mask] / dmax))))

        for i in range(Nx):
            for j in range(Ny - 1):
                px = i * dx
                py = (j + 0.5) * dy
                P = np.array([px, py])
                
                AP = P - A
                t = np.zeros(len(edges))
                t[valid] = np.sum(AP[valid] * AB[valid], axis=1) / AB_len_sq[valid]
                t = np.clip(t, 0.0, 1.0)
                proj = A + t[:, np.newaxis] * AB
                d = np.linalg.norm(P - proj, axis=1)
                
                mask = d < dmax
                ky_faces[i, j] = k_base * (1.0 + np.sum(1.0 / (1.0 + (d[mask] / dmax))))

        return kx_faces, ky_faces

    def resolver_sistema_ex1(self, kx_faces, ky_faces, Tc=35.0):
        Nx, Ny = self.placa.Nx, self.placa.Ny
        dx = self.placa.Lx / (Nx - 1)
        dy = self.placa.Ly / (Ny - 1)
        nunk = Nx * Ny

        rows, cols, data = [], [], []
        b = np.zeros(nunk)
        
        # Posição correta do círculo
        xc, yc = self.placa.Lx * 0.75, self.placa.Ly * 0.5
        R = self.placa.R

        for j in range(Ny):
            for i in range(Nx):
                Ic = i + j * Nx 
                
                x_coord = i * dx
                y_coord = j * dy
                dist_sq = (x_coord - xc)**2 + (y_coord - yc)**2

                if dist_sq <= R**2:
                    rows.append(Ic); cols.append(Ic); data.append(1.0)
                    b[Ic] = Tc

                elif i == 0:
                    rows.append(Ic); cols.append(Ic); data.append(1.0)
                    b[Ic] = 10.0
                    
                elif i == Nx - 1:
                    rows.append(Ic); cols.append(Ic); data.append(1.0)
                    b[Ic] = 30.0
                    
                elif j == 0 or j == Ny - 1:
                    rows.append(Ic); cols.append(Ic); data.append(1.0)
                    b[Ic] = 10.0 + 20.0 * (i / (Nx - 1))
                
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
                
                sistema = HidraulicoTermico.instantiate_subsystems(Nx, Ny)
                dx = sistema.placa.Lx / (Nx - 1)
                dy = sistema.placa.Ly / (Ny - 1)

                kx_faces, ky_faces = sistema.calcular_k_faces(dmax)
                T_array = sistema.resolver_sistema_ex1(kx_faces, ky_faces, Tc=35.0)
                
                sistema.placa.T = T_array
                T_grid = T_array.reshape((Ny, Nx)).T 
                
                tempo_total = time.perf_counter() - inicio

                # ==========================================
                # PLOTAGEM ESTILO REFERÊNCIA (Rede Direcionada)
                # ==========================================
                X_plot, Y_plot = np.meshgrid(np.linspace(0, sistema.placa.Lx, Nx), 
                                             np.linspace(0, sistema.placa.Ly, Ny), indexing='ij')
                
                fig, ax = plt.subplots(figsize=(10, 4.5))
                
                contorno = ax.contourf(X_plot, Y_plot, T_grid, 50, cmap='jet')
                ax.contour(X_plot, Y_plot, T_grid, 20, colors='k', linewidths=0.2)
                
                coord = sistema.rede.posicoes_nos
                edges = sistema.rede.conectividade
                
                # Plot das arestas com setas no meio
                for (n1, n2) in edges:
                    x1, y1 = coord[n1]
                    x2, y2 = coord[n2]
                    ax.plot([x1, x2], [y1, y2], color='black', linewidth=0.8, zorder=2)
                    
                    xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    vec_x, vec_y = x2 - x1, y2 - y1
                    ax.annotate('', xy=(xm + vec_x*0.05, ym + vec_y*0.05), xytext=(xm - vec_x*0.05, ym - vec_y*0.05),
                                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0), zorder=3)
                
                # Colore os nós como 'salmão' imitando o coolwarm da referência
                ax.scatter(coord[:, 0], coord[:, 1], facecolors='#ffb3b3', edgecolors='black', s=25, zorder=4)

                # Trava a janela do plot nas dimensões físicas da placa para não vazar
                ax.set_xlim(0, sistema.placa.Lx)
                ax.set_ylim(0, sistema.placa.Ly)
                ax.set_aspect('equal')
                
                ax.set_title(f"Temperatura e Rede Hidráulica - {Nx}x{Ny} | dmax={dmax}")
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')

                plt.colorbar(contorno, ax=ax, label="Temperatura (°C)", orientation='vertical', pad=0.02)
                plt.tight_layout()
                plt.show()

                # ==========================================
                # Perfis 1D
                # ==========================================
                # ==========================================
                # Perfis 1D (CORRIGIDO)
                # ==========================================
                # T_grid tem shape (Nx, Ny)
                # Fixa X no centro, varia Y -> Tamanho Ny (31)
                mid_vertical = T_grid[Nx // 2, :]  
                
                # Fixa Y no centro, varia X -> Tamanho Nx (61)
                mid_horizontal = T_grid[:, Ny // 2] 

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                
                # Plot Vertical: Y (tamanho Ny) vs T (tamanho Ny)
                ax1.plot(np.linspace(0, sistema.placa.Ly, Ny), mid_vertical)
                ax1.set_title(f"Perfil vertical (x={sistema.placa.Lx/2:.4f})")
                ax1.set_xlabel("y (m)")
                ax1.set_ylabel("T (°C)")
                ax1.grid(True)

                # Plot Horizontal: X (tamanho Nx) vs T (tamanho Nx)
                ax2.plot(np.linspace(0, sistema.placa.Lx, Nx), mid_horizontal)
                ax2.set_title(f"Perfil horizontal (y={sistema.placa.Ly/2:.4f})")
                ax2.set_xlabel("x (m)")
                ax2.grid(True)
                
                plt.tight_layout()
                plt.show()

                Tmax = np.max(T_grid)

                resultados.append({
                    "malha": f"{Nx}x{Ny}",
                    "dmax": dmax,
                    "Tmax": Tmax,
                    "tempo_total": tempo_total
                })

        return pd.DataFrame(resultados)
    

# =======================================================================
# FUNÇÕES DE EXECUÇÃO AUXILIARES
# =======================================================================

def ex_2_acoplamento():
    acoplamento = HidraulicoTermico.instantiate_subsystems(241, 121)
    for method in ['linear', 'nearest', 'cubic']:
        acoplamento.mapa_contorno_grade_secundaria(101, 51, method=method)
    
    acoplamento_reduzido = HidraulicoTermico.instantiate_subsystems(61, 31)
    for method in ['linear', 'nearest', 'cubic']:
        acoplamento_reduzido.mapa_contorno_grade_secundaria(41, 21, method=method)
    
    acoplamento.plotar_rede_termica(method='linear')


def ex_3_acoplamento():
    acoplamento = HidraulicoTermico.instantiate_subsystems(241, 121)
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
    malhas = [(31, 15), (121, 61), (241, 121)]
    configuracoes = [('ponto_medio', 10), ('trapezio', 10), ('trapezio', 100)]

    resultados = []

    for Nx, Ny in malhas:
        print(f"Processando malha: {Nx}x{Ny}...")
        sistema = HidraulicoTermico.instantiate_subsystems(Nx, Ny)
        
        sistema.atualizar_condutancias_ex4(metodo='trapezio', n_sub=10)
        
        for metodo, n_sub in configuracoes:
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
                'Potência (mW)': sistema.rede.calcular_potencia()/1000,
                'Tempo_Medio (s)': np.mean(tempos)
            })

    df = pd.DataFrame(resultados)
    pd.options.display.float_format = '{:.7e}'.format
    print("\n" + df.to_string(index=False))

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
    sistema = HidraulicoTermico.instantiate_subsystems(241, 121) 
    
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
    
    sim_base = HidraulicoTermico.instantiate_subsystems(Nx, Ny)
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
            
            Sp = calcular_termo_fonte_gaussiano(Nx, Ny, Lx, Ly, coord, edges, S0, dist, d_max)
            fonte_total = 5e5 + Sp
            
            placa_modificada = PlacaTermica(
                Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, k=0.25, R=0.0025,
                fonte_calor=fonte_total.T  
            )
            placa_modificada.resolver_circulo(Tc=35, mode='sparse')
            
            interpolador = placa_modificada.criar_interpolador('linear')
            x_g = np.linspace(0.0, Lx, Nx)
            y_g = np.linspace(0.0, Ly, Ny)
            Xs, Ys = np.meshgrid(x_g, y_g, indexing='ij')
            pts_all = np.column_stack([Xs.ravel(), Ys.ravel()])
            T_all = interpolador(pts_all)
            
            T_max = np.max(T_all)
            temperaturas_maximas[dist].append((S0, T_max))
            
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
            
            x_perf = np.linspace(0.0, Lx, 300)
            y_perf = np.ones_like(x_perf) * (Ly / 2.0)
            T_horiz = interpolador(np.column_stack([x_perf, y_perf]))
            perfis_horizontais.append((S0, T_horiz))
            
            y_perf_v = np.linspace(0.0, Ly, 300)
            x_perf_v = np.ones_like(y_perf_v) * (Lx / 2.0)
            T_vert = interpolador(np.column_stack([x_perf_v, y_perf_v]))
            perfis_verticais.append((S0, T_vert))
            
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


def ex_1_especial_acoplamento():
    print("\n--- INICIANDO ROTINA ESPECIAL: MAPA DE PROXIMIDADE E CONDUTIVIDADE ---")
    sistema_gatilho = HidraulicoTermico(61, 31)
    print("Iniciando varredura de malhas e cálculo de distâncias (isso pode demorar devido aos laços não-vetorizados)...")
    df_resultados = sistema_gatilho.exercicio_1_2()
    print("\n=======================================================")
    print("RESULTADOS GLOBAIS: TEMPERATURA MÁXIMA vs DISTÂNCIA (dmax)")
    print("=======================================================")
    print(df_resultados.to_string(index=False))

if __name__ == "__main__":
    sistema_teste = HidraulicoTermico.instantiate_subsystems(61, 31)
    sistema_teste.exercicio_1_2()