import numpy as np
import matplotlib.pyplot as plt

import time

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

        self.placa.resolver_circulo(
            Tc=35,
            mode='sparse'
        )

        self.rede = RedeHidraulica(
            levels=3
        )

    def temperatura_aresta(
        self,
        i,
        j,
        interpolador,
        metodo='trapezio',
        n_sub=1
    ):
        p0 = self.rede.posicoes_nos[i]
        p1 = self.rede.posicoes_nos[j]

        def ponto(t):
            return (1 - t) * p0 + t * p1

        if metodo == 'ponto_medio':
            t = 0.5
            return interpolador([ponto(t)])[0]

        ts = np.linspace(0.0, 1.0, n_sub + 1)

        valores = np.array([
            interpolador([ponto(t)])[0]
            for t in ts
        ])

        h = 1.0 / n_sub

        if metodo == 'trapezio':
            integral = (
                h *
                (
                    0.5 * valores[0]
                    + np.sum(valores[1:-1])
                    + 0.5 * valores[-1]
                )
            )

            return integral

        elif metodo == 'simpson':
            if n_sub % 2 == 1:
                n_sub += 1

                ts = np.linspace(0.0, 1.0, n_sub + 1)

                valores = np.array([
                    interpolador([ponto(t)])[0]
                    for t in ts
                ])

                h = 1.0 / n_sub

            integral = (
                h / 3.0 *
                (
                    valores[0]
                    + valores[-1]
                    + 4 * np.sum(valores[1:-1:2])
                    + 2 * np.sum(valores[2:-2:2])
                )
            )

            return integral

        raise ValueError("Método inválido")
    
    def temperaturas_medias_arestas(
        self,
        metodo='trapezio',
        n_sub=1,
        interp='linear'
    ):
        interpolador = self.placa.criar_interpolador(interp)

        temperaturas = []

        inicio = time.perf_counter()

        for (i, j) in self.rede.conectividade:
            T_med = self.temperatura_aresta(
                i,
                j,
                interpolador,
                metodo=metodo,
                n_sub=n_sub
            )

            temperaturas.append(T_med)

        fim = time.perf_counter()

        temperaturas = np.array(temperaturas)

        tempo = fim - inicio

        return temperaturas, tempo
    
    def plotar_temperatura_arestas(
        self,
        temperaturas
    ):
        coord = self.rede.posicoes_nos
        edges = self.rede.conectividade

        fig, ax = plt.subplots(figsize=(10, 5))

        cmap = plt.get_cmap('jet')

        norm = plt.Normalize(
            temperaturas.min(),
            temperaturas.max()
        )

        for k, (i, j) in enumerate(edges):
            x1, y1 = coord[i]
            x2, y2 = coord[j]

            cor = cmap(norm(temperaturas[k]))

            ax.plot(
                [x1, x2],
                [y1, y2],
                color=cor,
                linewidth=3
            )

        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=norm
        )

        plt.colorbar(
            sm,
            ax=ax,
            label='Temperatura média'
        )

        ax.set_aspect('equal')

        plt.show()

    def temperaturas_nos(self, method='linear'):
        coords = self.rede.posicoes_nos

        interpolador = self.placa.criar_interpolador(method)

        temperaturas = interpolador(coords)

        return temperaturas

    def atualizar_rede(self, method='linear'):
        temperaturas = self.temperaturas_nos(method)

        self.rede.atualizar_condutancias(
            temperaturas
        )

        self.rede.resolver()

        return temperaturas

    def plotar_rede_termica(self, method='linear'):
        temperaturas = self.atualizar_rede(method)

        coord = self.rede.posicoes_nos
        edges = self.rede.conectividade

        fig, ax = plt.subplots(figsize=(10, 8))

        for (i, j) in edges:
            x1, y1 = coord[i]
            x2, y2 = coord[j]

            ax.plot(
                [x1, x2],
                [y1, y2],
                color='black',
                linewidth=1.5,
                zorder=1
            )

        scatter = ax.scatter(
            coord[:,0],
            coord[:,1],
            c=temperaturas,
            cmap='jet',
            s=250,
            edgecolors='black',
            zorder=2
        )

        for i, (x, y) in enumerate(coord):
            ax.text(
                x,
                y,
                f'{i+1}',
                ha='center',
                va='center',
                fontweight='bold',
                color='white'
            )

        plt.colorbar(
            scatter,
            ax=ax,
            label='Temperatura (°C)'
        )

        ax.set_aspect('equal')

        ax.set_title(
            f'Temperatura nos nós ({method})'
        )

        plt.show()

    def mapa_contorno_grade_secundaria(
        self,
        Nx_sec,
        Ny_sec,
        method='linear'
    ):
        x_sec = np.linspace(
            0.0,
            self.placa.Lx,
            Nx_sec
        )

        y_sec = np.linspace(
            0.0,
            self.placa.Ly,
            Ny_sec
        )

        Xs, Ys = np.meshgrid(
            x_sec,
            y_sec,
            indexing='ij'
        )

        pts = np.column_stack([
            Xs.ravel(),
            Ys.ravel()
        ])

        interpolador = self.placa.criar_interpolador(method)

        T_sec = interpolador(pts).reshape(
            Nx_sec,
            Ny_sec
        )

        fig, ax = plt.subplots(figsize=(8,4))

        cont = ax.contourf(
            Xs,
            Ys,
            T_sec,
            20,
            cmap='jet'
        )

        ax.contour(
            Xs,
            Ys,
            T_sec,
            20,
            colors='k',
            linewidths=0.3
        )

        ax.set_aspect('equal')

        ax.set_title(
            f'Interpolação {method} ({Nx_sec}x{Ny_sec})'
        )

        plt.colorbar(cont, ax=ax)

        plt.show()

def ex_2_acoplamento():
    acoplamento = HidraulicoTermico(241, 121)

    acoplamento.mapa_contorno_grade_secundaria(
        101,
        51,
        method='linear'
    )

    acoplamento.mapa_contorno_grade_secundaria(
        101,
        51,
        method='nearest'
    )

    acoplamento.mapa_contorno_grade_secundaria(
        101,
        51,
        method='cubic'
    )

    acoplamento = HidraulicoTermico(61, 31)

    acoplamento.mapa_contorno_grade_secundaria(
        41,
        21,
        method='linear'
    )

    acoplamento.mapa_contorno_grade_secundaria(
        41,
        21,
        method='nearest'
    )

    acoplamento.mapa_contorno_grade_secundaria(
        41,
        21,
        method='cubic'
    )

    acoplamento.plotar_rede_termica(
        method='linear'
    )

def ex_3_acoplamento():
    acoplamento = HidraulicoTermico(241, 121)

    configs = [
        ('trapezio', 1),
        ('trapezio', 10),
        ('trapezio', 100),
        ('trapezio', 1000),
        ('simpson', 10),
        ('simpson', 100),
        ('simpson', 1000)
    ]

    for metodo, n in configs:
        Tmed, tempo = acoplamento.temperaturas_medias_arestas(
            metodo=metodo,
            n_sub=n
        )

        print()
        print(f'Método: {metodo}')
        print(f'Subdivisões: {n}')
        print(f'Tempo: {tempo:.6f} s')
        print(f'Temperatura média global: {Tmed.mean():.6f}')

    Tmed, tempo = acoplamento.temperaturas_medias_arestas(
        metodo='simpson',
        n_sub=100
    )

    acoplamento.plotar_temperatura_arestas(Tmed)


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
