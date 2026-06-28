import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.graphs_utils.gera_grafo import gera_grafo

class RedeHidraulica:
    def __init__(self, levels:int=3, A_k=2.5e-7):
        """Construtor da rede hidráulica acoplada à placa térmica."""

        self.Lx = 0.03
        self.Ly = 0.015

        self.A_k = A_k

        self.D_k = np.sqrt(
            4 * self.A_k / np.pi
        )

        self.temperatura_referencia = 20.0

        coordenadas, conectividade = gera_grafo(levels)
        coordenadas *= 1e-3

        self.numero_nos = len(coordenadas)

        self.conectividade = np.array(
            conectividade
        )

        self.posicoes_nos = np.array(
            coordenadas
        )

        self.condutancias = self._calcular_condutancias(
            np.full(
                self.numero_nos,
                self.temperatura_referencia
            )
        )

        self.vazoes_por_no = None
        self.pressao_por_no = None

        self.matriz_global = None

        self.pressao = None
        self.vazao = None

        self.historico_pressao = []
        self.historico_vazao = []

    def viscosidade(self, T):
        return (
            0.001791 /
            (
                1
                + 0.03368 * T
                + 0.000221 * T**2
            )
        )

    def _calcular_condutancias(
        self,
        temperaturas
    ):
        condutancias = []

        for (i, j) in self.conectividade:
            T_media = 0.5 * (
                temperaturas[i]
                + temperaturas[j]
            )

            mu = self.viscosidade(
                T_media
            )

            kappa_k = (
                np.pi * self.D_k**4
            ) / (
                128 * mu
            )

            no_1 = self.posicoes_nos[i]
            no_2 = self.posicoes_nos[j]

            L_k = np.sqrt(
                (no_1[0] - no_2[0])**2
                +
                (no_1[1] - no_2[1])**2
            )

            condutancia = (
                kappa_k / L_k
            )

            condutancias.append(
                condutancia
            )

        return np.array(
            condutancias
        )

    def assembly(self):
        """Monta a matriz global do sistema acumulando as matrizes locais de cada cano."""
        self.matriz_global = np.zeros((self.numero_nos, self.numero_nos))

        for k, (idx_i, idx_j) in enumerate(self.conectividade):
            ck = self.condutancias[k]

            self.matriz_global[idx_i, idx_i] += ck
            self.matriz_global[idx_j, idx_j] += ck

            self.matriz_global[idx_i, idx_j] -= ck
            self.matriz_global[idx_j, idx_i] -= ck

    def resolver(
    self,
    pressao_imposta=None,
    vazao_imposta=None
    ):
        """Resolve a rede utilizando as condições do problema."""

        if pressao_imposta is None:
            pressao_imposta = {}

        if vazao_imposta is None:
            vazao_imposta = {}

        if self.matriz_global is None:
            self.assembly()

        matriz_modificada = self.matriz_global.copy()

        vazao_modificada = np.zeros(
            self.numero_nos
        )

        # Condições extras opcionais
        for k, vazao in vazao_imposta.items():
            vazao_modificada[k - 1] = vazao

        for k, pressao in pressao_imposta.items():
            matriz_modificada[k - 1, :] = 0
            matriz_modificada[k - 1, k - 1] = 1

            vazao_modificada[k - 1] = pressao

        self.pressao = np.linalg.solve(
            matriz_modificada,
            vazao_modificada
        )

        self.calcular_vazoes()

        self.historico_pressao.append(
            self.pressao.copy()
        )

        self.historico_vazao.append(
            self.vazao.copy()
        )

        return self.pressao

    def calcular_vazoes(self):
        """Calcula as vazões nos canos usando a fórmula Q = KDp."""

        numero_canos = len(self.conectividade)

        matriz_incidencia = np.zeros(
            (numero_canos, self.numero_nos)
        )

        for k, (i, j) in enumerate(self.conectividade):
            matriz_incidencia[k, i] = 1
            matriz_incidencia[k, j] = -1
        
        matriz_condutancias = np.diag(
            self.condutancias
        )

        self.vazao = (
            matriz_condutancias
            @ matriz_incidencia
            @ self.pressao
        )
        
        return self.vazao
    
    def calcular_potencia(self):
        """Calcula a potência consumida na rede empregando a expressão W = p^T(D^TKD)p"""

        if self.pressao is None:
            print("Erro: Resolva a rede antes de plotar.")
            return None

        numero_canos = len(self.conectividade)

        matriz_incidencia = np.zeros(
            (numero_canos, self.numero_nos)
        )

        for k, (i, j) in enumerate(self.conectividade):
            matriz_incidencia[k, i] = 1
            matriz_incidencia[k, j] = -1
        
        matriz_condutancias = np.diag(
            self.condutancias
        )

        W = (
            self.pressao.T
            @ matriz_incidencia.T
            @ matriz_condutancias
            @ matriz_incidencia
            @ self.pressao
        )

        return W

    def plotaRede(self, scale=1.0, save_path=None):
        """Abre uma janela do matplotlib que plota o grafo."""

        if self.pressao is None or self.vazao is None:
            print("Erro: Resolva a rede antes de plotar.")
            return

        coord = self.posicoes_nos * scale

        edges = self.conectividade

        p = self.pressao
        q = self.vazao

        segs = []
        mids = []

        for (i, j) in edges:
            x1, y1 = coord[i, 0], coord[i, 1]
            x2, y2 = coord[j, 0], coord[j, 1]

            segs.append(((x1, y1), (x2, y2)))

            mids.append(
                (
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0
                )
            )

        segs = np.array(segs)
        mids = np.array(mids)

        fig, ax = plt.subplots(figsize=(10, 10))

        cmap = plt.get_cmap("coolwarm")

        norm = plt.Normalize(
            vmin=float(p.min()),
            vmax=float(p.max())
        )
        
        colors = [cmap(norm(pi)) for pi in p]

        ax.scatter(
            coord[:, 0],
            coord[:, 1],
            s=500,
            c=colors,
            zorder=3,
            edgecolors="black"
        )

        arrow_scale = 0.05

        for idx, ((x1, y1), (x2, y2)) in enumerate(segs):
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="black",
                linewidth=2.0,
                zorder=1
            )
            
            xm, ym = mids[idx]

            dx, dy = x2 - x1, y2 - y1

            L = np.hypot(dx, dy)

            if L == 0:
                continue
            
            dxn, dyn = dx / L, dy / L

            nx, ny = -dyn, dxn
            
            q_dir = 1 if p[edges[idx, 0]] > p[edges[idx, 1]] else -1

            ax.annotate(
                "",
                xy=(
                    xm + q_dir * 1.5 * arrow_scale * dxn,
                    ym + q_dir * 1.5 * arrow_scale * dyn
                ),
                xytext=(
                    xm - q_dir * 1.5 * arrow_scale * dxn,
                    ym - q_dir * 1.5 * arrow_scale * dyn
                ),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="black",
                    lw=1.5,
                    mutation_scale=20
                ),
                zorder=5
            )

            ax.text(
                xm + nx * 0.1,
                ym + ny * 0.1,
                f"q={q[idx]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                zorder=6,
                bbox=dict(
                    facecolor='white',
                    alpha=0.7,
                    edgecolor='none'
                )
            )

        for i, (x, y) in enumerate(coord):
            ax.text(
                x,
                y,
                str(i+1),
                ha="center",
                va="center",
                fontweight='bold',
                zorder=4
            )

            ax.text(
                x,
                y - 0.15,
                f"p={p[i]:.2f}",
                ha="center",
                va="top",
                fontsize=9,
                color="blue"
            )

        ax.set_aspect("equal")

        ax.axis("off")

        ax.set_xlim(
            coord[:,0].min() - 0.5,
            coord[:,0].max() + 0.5
        )

        ax.set_ylim(
            coord[:,1].min() - 0.5,
            coord[:,1].max() + 0.5
        )

        sm = cm.ScalarMappable(
            cmap=cmap,
            norm=norm
        )

        plt.colorbar(
            sm,
            ax=ax,
            label="Pressão (p)"
        )

        if save_path:
            plt.savefig(save_path, dpi=300)

        plt.show()

    def atualizar_condutancias(
        self,
        temperaturas
    ):
        self.condutancias = (
            self._calcular_condutancias(
                temperaturas
            )
        )

        self.matriz_global = None


def calcular_potencias_bombas(
    bombas:dict,
    rede:RedeHidraulica,
    cenario_index:int=-1,
    p_noatm:float=0.0
):
    if not rede.historico_pressao:
        raise ValueError(
            "A rede precisa estar resolvida para calcular a potência das bombas"
        )

    pressoes_cenario = rede.historico_pressao[cenario_index]

    potencia_total = 0.0

    potencias_individuais = {}

    for no_bomba, q_bomba in bombas.items():
        index_bomba = no_bomba - 1

        p_bomba = pressoes_cenario[index_bomba]

        potencia = (
            (p_bomba - p_noatm)
            * q_bomba
        )

        potencias_individuais[no_bomba] = potencia

        potencia_total += potencia

    return potencia_total, potencias_individuais