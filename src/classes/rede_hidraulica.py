import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class RedeHidraulica:
    def __init__(self, n_nos, conectividade, condutancias, coordenadas=None):
        self.n_nos = n_nos
        # Armazenamos a conectividade em base 0 internamente para facilitar o plot
        self.conec = np.array(conectividade) - 1 
        self.C = np.array(condutancias)
        self.Xno = np.array(coordenadas) if coordenadas is not None else None
        self.A = np.zeros((n_nos, n_nos))
        self.p = None
        self.q = None

    def assembly(self):
        """Monta a matriz global A acumulando as matrizes locais de cada cano"""
        self.A = np.zeros((self.n_nos, self.n_nos))
        for k, (idx_i, idx_j) in enumerate(self.conec):
            ck = self.C[k]
            self.A[idx_i, idx_i] += ck
            self.A[idx_j, idx_j] += ck
            self.A[idx_i, idx_j] -= ck
            self.A[idx_j, idx_i] -= ck

    def resolver(self, no_atm, no_bomba, q_bomba):
        """Resolve o sistema linear modificando a linha do nó atmosférico[cite: 217, 492]."""
        Atilde = self.A.copy()
        b = np.zeros(self.n_nos)
        
        # Condição de contorno: Pressão fixada (p_atm = 0)
        idx_atm = no_atm - 1
        Atilde[idx_atm, :] = 0
        Atilde[idx_atm, idx_atm] = 1
        
        # Fonte: Vazão da bomba injetada no nó nB
        b[no_bomba - 1] = q_bomba
        
        self.p = np.linalg.solve(Atilde, b)
        self.calcular_vazoes()
        return self.p

    def calcular_vazoes(self):
        """Calcula as vazões nos canos usando a fórmula Q = K * D * p."""
        nc = len(self.conec)
        # Matriz de incidência D
        D = np.zeros((nc, self.n_nos))
        for k, (i, j) in enumerate(self.conec):
            D[k, i] = 1
            D[k, j] = -1
        
        # Matriz diagonal de condutâncias K
        K = np.diag(self.C)
        self.q = K @ D @ self.p
        return self.q

    def plotaRede(self, save_path=None):
        """Abre uma janela do matplotlib que plota o grafo."""
        if self.p is None or self.q is None:
            print("Erro: Resolva a rede antes de plotar.")
            return

        coord = self.Xno
        if coord is None:
            print("Erro: Coordenadas dos nós não fornecidas no construtor.")
            return

        edges = self.conec
        p = self.p
        q = self.q
        nv = self.n_nos

        segs = []
        mids = []
        for (i, j) in edges:
            x1, y1 = coord[i, 0], coord[i, 1]
            x2, y2 = coord[j, 0], coord[j, 1]
            segs.append(((x1, y1), (x2, y2)))
            mids.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

        segs = np.array(segs)
        mids = np.array(mids)

        fig, ax = plt.subplots(figsize=(10, 10))
        cmap = plt.get_cmap("coolwarm")
        norm = plt.Normalize(vmin=float(p.min()), vmax=float(p.max()))
        
        # Plot dos nós
        colors = [cmap(norm(pi)) for pi in p]
        ax.scatter(coord[:, 0], coord[:, 1], s=500, c=colors, zorder=3, edgecolors="black")

        # Plot das arestas e setas de fluxo
        arrow_scale = 0.05
        for idx, ((x1, y1), (x2, y2)) in enumerate(segs):
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=2.0, zorder=1)
            
            xm, ym = mids[idx]
            dx, dy = x2 - x1, y2 - y1
            L = np.hypot(dx, dy)
            if L == 0: continue
            
            dxn, dyn = dx / L, dy / L
            nx, ny = -dyn, dxn
            
            # Direção da seta baseada na pressão (do maior para o menor)
            q_dir = 1 if p[edges[idx, 0]] > p[edges[idx, 1]] else -1

            ax.annotate("", 
                xy=(xm + q_dir * 1.5 * arrow_scale * dxn, ym + q_dir * 1.5 * arrow_scale * dyn),
                xytext=(xm - q_dir * 1.5 * arrow_scale * dxn, ym - q_dir * 1.5 * arrow_scale * dyn),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=20),
                zorder=5)

            # Rótulo da vazão
            ax.text(xm + nx * 0.1, ym + ny * 0.1, f"q={q[idx]:.2f}", 
                    ha="center", va="center", fontsize=10, zorder=6,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Rótulos dos nós
        for i, (x, y) in enumerate(coord):
            ax.text(x, y, str(i+1), ha="center", va="center", fontweight='bold', zorder=4)
            ax.text(x, y - 0.15, f"p={p[i]:.2f}", ha="center", va="top", fontsize=9, color="blue")

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(coord[:,0].min() - 0.5, coord[:,0].max() + 0.5)
        ax.set_ylim(coord[:,1].min() - 0.5, coord[:,1].max() + 0.5)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label="Pressão (p)")
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label="Pressão (p)")
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

def calcular_potencia_bomba(q_bomba,no_bomba,rede,p_noatm):
    pbomba = rede.p[no_bomba - 1]
    return (pbomba-p_noatm)*q_bomba

