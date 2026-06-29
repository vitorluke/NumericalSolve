import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.graphs_utils.gera_grafo import gera_grafo


class RedeHidraulica:
    def __init__(self, levels: int = 3, A_k=None, H_k=None):

        assert (A_k is not None) or (H_k is not None)

        self.Lx = 0.03
        self.Ly = 0.015

        if A_k is not None:
            self.A_k = A_k
            self.D_k = np.sqrt(
                4.0 * A_k / np.pi
            )
        else:
            self.D_k = H_k

        self.temperatura_referencia = 20.0

        coordenadas, conectividade = gera_grafo(levels)
        coordenadas *= 1e-3

        self.numero_nos = len(coordenadas)

        self.conectividade = np.asarray(
            conectividade,
            dtype=np.int32
        )

        self.posicoes_nos = np.asarray(
            coordenadas,
            dtype=np.float64
        )

        self.i_edges = self.conectividade[:, 0]
        self.j_edges = self.conectividade[:, 1]

        dx = (
            self.posicoes_nos[self.i_edges, 0]
            - self.posicoes_nos[self.j_edges, 0]
        )

        dy = (
            self.posicoes_nos[self.i_edges, 1]
            - self.posicoes_nos[self.j_edges, 1]
        )

        self.comprimentos = np.sqrt(
            dx * dx + dy * dy
        )

        self.numero_canos = len(
            self.conectividade
        )

        rows = np.repeat(
            np.arange(self.numero_canos),
            2
        )

        cols = self.conectividade.reshape(-1)

        data = np.tile(
            [1.0, -1.0],
            self.numero_canos
        )

        self.matriz_incidencia = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(
                self.numero_canos,
                self.numero_nos
            )
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
            0.001791
            /
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
        T_media = 0.5 * (
            temperaturas[self.i_edges]
            + temperaturas[self.j_edges]
        )

        mu = self.viscosidade(
            T_media
        )

        kappa = (
            np.pi * self.D_k**4
        ) / (
            128.0 * mu
        )

        return (
            kappa
            / self.comprimentos
        )

    def assembly(self):

        ne = self.numero_canos

        rows = np.empty(
            4 * ne,
            dtype=np.int32
        )

        cols = np.empty(
            4 * ne,
            dtype=np.int32
        )

        data = np.empty(
            4 * ne,
            dtype=np.float64
        )

        rows[0::4] = self.i_edges
        rows[1::4] = self.j_edges
        rows[2::4] = self.i_edges
        rows[3::4] = self.j_edges

        cols[0::4] = self.i_edges
        cols[1::4] = self.j_edges
        cols[2::4] = self.j_edges
        cols[3::4] = self.i_edges

        data[0::4] = self.condutancias
        data[1::4] = self.condutancias
        data[2::4] = -self.condutancias
        data[3::4] = -self.condutancias

        self.matriz_global = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(
                self.numero_nos,
                self.numero_nos
            )
        ).tocsr()

    def resolver(
        self,
        pressao_imposta=None,
        vazao_imposta=None
    ):

        if pressao_imposta is None:
            pressao_imposta = {}

        if vazao_imposta is None:
            vazao_imposta = {}

        if self.matriz_global is None:
            self.assembly()

        matriz_modificada = (
            self.matriz_global
            .tolil(copy=True)
        )

        rhs = np.zeros(
            self.numero_nos
        )

        for k, vazao in vazao_imposta.items():
            rhs[k - 1] = vazao

        for k, pressao in pressao_imposta.items():

            idx = k - 1

            matriz_modificada[idx, :] = 0.0
            matriz_modificada[idx, idx] = 1.0

            rhs[idx] = pressao

        self.pressao = spla.spsolve(
            matriz_modificada.tocsc(),
            rhs
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

        dp = (
            self.matriz_incidencia
            @ self.pressao
        )

        self.vazao = (
            self.condutancias
            * dp
        )

        return self.vazao

    def calcular_potencia(self):

        if self.pressao is None:
            print(
                "Erro: Resolva a rede antes de plotar."
            )
            return None

        return (
            self.pressao
            @ (
                self.matriz_global
                @ self.pressao
            )
        )

    def plotaRede(
        self,
        scale=1.0,
        save_path=None
    ):

        if (
            self.pressao is None
            or self.vazao is None
        ):
            print(
                "Erro: Resolva a rede antes de plotar."
            )
            return

        coord = (
            self.posicoes_nos
            * scale
        )

        edges = self.conectividade

        p = self.pressao
        q = self.vazao

        segs = []
        mids = []

        for (i, j) in edges:

            x1, y1 = coord[i]
            x2, y2 = coord[j]

            segs.append(
                (
                    (x1, y1),
                    (x2, y2)
                )
            )

            mids.append(
                (
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0
                )
            )

        segs = np.asarray(segs)
        mids = np.asarray(mids)

        fig, ax = plt.subplots(
            figsize=(10, 10)
        )

        cmap = plt.get_cmap(
            "coolwarm"
        )

        norm = plt.Normalize(
            vmin=float(p.min()),
            vmax=float(p.max())
        )

        colors = [
            cmap(norm(pi))
            for pi in p
        ]

        ax.scatter(
            coord[:, 0],
            coord[:, 1],
            s=500,
            c=colors,
            zorder=3,
            edgecolors="black"
        )

        arrow_scale = 0.05

        for idx, (
            (x1, y1),
            (x2, y2)
        ) in enumerate(segs):

            ax.plot(
                [x1, x2],
                [y1, y2],
                color="black",
                linewidth=2.0,
                zorder=1
            )

            xm, ym = mids[idx]

            dx = x2 - x1
            dy = y2 - y1

            L = np.hypot(dx, dy)

            if L == 0:
                continue

            dxn = dx / L
            dyn = dy / L

            nx = -dyn
            ny = dxn

            q_dir = (
                1
                if p[edges[idx, 0]]
                > p[edges[idx, 1]]
                else -1
            )

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
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none"
                )
            )

        for i, (x, y) in enumerate(coord):

            ax.text(
                x,
                y,
                str(i + 1),
                ha="center",
                va="center",
                fontweight="bold",
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
            coord[:, 0].min() - 0.5,
            coord[:, 0].max() + 0.5
        )

        ax.set_ylim(
            coord[:, 1].min() - 0.5,
            coord[:, 1].max() + 0.5
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
            plt.savefig(
                save_path,
                dpi=300
            )

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