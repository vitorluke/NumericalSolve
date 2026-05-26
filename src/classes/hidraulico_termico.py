import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

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
        self.placa.resolver_circulo(Tc=35, mode='sparse')
        self.rede = RedeHidraulica(levels=3)

    def calcular_viscosidade(self, T):
        return 0.001791 / (1.0 + 0.03368 * T + 0.000221 * (T**2))

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
        ('monte_carlo', 10),
        ('monte_carlo', 100),
        ('ponto_medio', 10),
        ('ponto_medio', 100),
        ('trapezio', 1),
        ('trapezio', 10),
        ('trapezio', 100)
    ]

    for metodo, n in configs:
        Tmed, tempo = acoplamento.temperaturas_medias_arestas(metodo=metodo, n_sub=n)
        print(f'\nMétodo: {metodo}\nSubdivisões: {n}\nTempo: {tempo:.6f} s\nTemperatura média global: {Tmed.mean():.6f}')

    Tmed_plot, _ = acoplamento.temperaturas_medias_arestas(metodo='trapezio', n_sub=100)
    acoplamento.plotar_dados_arestas(Tmed_plot, label='Temperatura Média (°C)')

def ex_4_acoplamento():
    print("\n--- EXERCÍCIO 4: ANÁLISE DE CONVERGÊNCIA (MALHAS E QUADRATURA) ---")
    malhas = [(61, 31), (121, 61), (241, 121)]
    configuracoes = [('ponto_medio', 10), ('trapezio', 10), ('trapezio', 100)]
    
    resultados = []

    for Nx, Ny in malhas:
        sistema = HidraulicoTermico(Nx, Ny)
        for metodo, n_sub in configuracoes:
            inicio = time.perf_counter()
            T_med = sistema.atualizar_condutancias_ex4(metodo=metodo, n_sub=n_sub)
            fim = time.perf_counter()
            
            P_max = sistema.rede.pressao.max()
            Pot = sistema.rede.calcular_potencia()
            
            resultados.append({
                'Malha': f'{Nx}x{Ny}',
                'Método': metodo,
                'Subdivisões': n_sub,
                'P_max (Pa)': P_max,
                'Potência (W)': Pot,
                'Tempo (s)': fim - inicio
            })
            
    df = pd.DataFrame(resultados)
    print(df.to_string(index=False))

    sistema_plot = HidraulicoTermico(241, 121)
    T_med_plot = sistema_plot.atualizar_condutancias_ex4(metodo='trapezio', n_sub=100)
    sistema_plot.plotar_dados_arestas(T_med_plot, label='Temperatura Média (°C)')
    sistema_plot.plotar_rede_termica(method='linear')


def ex_4_convergencia_grafica():
    malha = (241, 121) # Fixando a malha mais refinada como referência
    subdivisoes = [1, 2, 5, 10, 20, 50, 100]
    
    sistema = HidraulicoTermico(malha[0], malha[1])
    
    pot_pm = []
    pot_trap = []
    
    print("Calculando convergência...")
    for n in subdivisoes:
        sistema.atualizar_condutancias_ex4(metodo='ponto_medio', n_sub=n)
        pot_pm.append(sistema.rede.calcular_potencia())
        
        sistema.atualizar_condutancias_ex4(metodo='trapezio', n_sub=n)
        pot_trap.append(sistema.rede.calcular_potencia())
        
    # Plotagem do Gráfico de Convergência
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(subdivisoes, pot_pm, marker='o', linestyle='-', color='blue', label='Ponto Médio', linewidth=2)
    ax.plot(subdivisoes, pot_trap, marker='s', linestyle='--', color='red', label='Trapézio', linewidth=2)
    
    ax.set_xscale('log')
    ax.set_xlabel('Número de Subdivisões por Aresta (escala log)')
    ax.set_ylabel('Potência Total Dissipada (W)')
    ax.set_title('Convergência da Potência Dissipada no Gêmeo Digital')
    
    # Adicionando grid para facilitar a leitura da estabilização
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend()
    
    plt.tight_layout()


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

    if diff_P < 1.0:
        print("-> Conclusão: A não-linearidade da função viscosidade gera um desvio baixo. A aproximação mu(<T>) é fisicamente aceitável para este gradiente térmico.")
    else:
        print("-> Conclusão: O desvio é significativo. O acoplamento exige a integração exata <mu(T)> para garantir precisão no Gêmeo Digital.")

    sistema.plotar_dados_arestas(viscosidades_efetivas, label='Viscosidade Efetiva')
    sistema.plotar_rede_termica(method='linear')

    ########################################################################
    # DISTÂNCIA ENTRE PONTO E SEGMENTO
    ########################################################################

def distancia_ponto_segmento(
     self,
     p,
    a,
    b
):
    ab = b - a

    ap = p - a

    t = np.dot(ap, ab) / np.dot(ab, ab)

    t = np.clip(t, 0.0, 1.0)

    proj = a + t * ab

    return np.linalg.norm(p - proj)

    ########################################################################
    # MAPA DE PROXIMIDADE
    ########################################################################

def criar_mapa_proximidade(
    self,
    dmax
):
    mapa = {}

    Nx = self.placa.Nx
    Ny = self.placa.Ny

    for i in range(Nx):

        for j in range(Ny):

            kglobal = i + j * Nx

            x = self.placa.X[i, j]
            y = self.placa.Y[i, j]

            p = np.array([x, y])

            vizinhos = []

            for edge_id, (n1, n2) in enumerate(
                self.rede.conectividade
            ):

                a = self.rede.posicoes_nos[n1]
                b = self.rede.posicoes_nos[n2]

                d = self.distancia_ponto_segmento(
                    p,
                    a,
                    b
                )

                    ################################################################
                    # SE A DISTÂNCIA FOR MENOR QUE dmax
                    # O CANAL INFLUENCIA O PONTO
                    ################################################################

                if d < dmax:

                    vizinhos.append(
                        (edge_id, d)
                    )

            mapa[kglobal] = vizinhos

    return mapa

    ########################################################################
    # CONDUTIVIDADE MODIFICADA
    ########################################################################

def k_interface(
    self,
    p,
    mapa
):
    Nx = self.placa.Nx
    Ny = self.placa.Ny

    dx = self.placa.dx
    dy = self.placa.dy

    x, y = p

        ####################################################################
        # CONVERTE COORDENADA FÍSICA EM ÍNDICE DA MALHA
        ####################################################################

    i = int(round(x / dx))
    j = int(round(y / dy))

    i = np.clip(i, 0, Nx - 1)
    j = np.clip(j, 0, Ny - 1)

    kglobal = i + j * Nx

        ####################################################################
        # PEGA TODOS OS CANAIS PRÓXIMOS
        ####################################################################

    vizinhos = mapa[kglobal]

        ####################################################################
        # SOMA AS CONTRIBUIÇÕES DAS DISTÂNCIAS
        ####################################################################

    soma = 0.0

    for edge_id, d in vizinhos:

        soma += 1.0 / (1.0 + d)

        ####################################################################
        # CONDUTIVIDADE MODIFICADA
        ####################################################################

    return self.placa.k * (
        1.0 + soma
    )

    ########################################################################
    # INICIALIZA O MAPA DE PROXIMIDADE
    ########################################################################

def inicializar_proximidade(
    self,
    dmax
):
    self.mapa_proximidade = (
        self.criar_mapa_proximidade(
            dmax
        )
    )



