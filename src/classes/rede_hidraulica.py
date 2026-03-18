import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class RedeHidraulica:
    def __init__(self, n_nos, conectividade, condutancias, coordenadas:None, vazao_por_no = None, pressao_por_no = None):
        """Construtor da classe da rede hidráulica."""
        self.numero_nos = n_nos
        
        self.conectividade = np.array(conectividade)
        self.condutancias = np.array(condutancias)
        self.posicoes_nos = np.array(coordenadas) if coordenadas is not None else None
        self.vazoes_por_no = vazao_por_no
        self.pressao_por_no = pressao_por_no

        self.matriz_global = None
        self.pressao = None
        self.vazao = None

        self.historico_pressao = []
        self.historico_vazao = []

        self.historico_pressao.append(self.pressao.copy())
        self.historico_vazao.append(self.vazao.copy())

    def assembly(self):
        """Monta a matriz global do sistema acumulando as matrizes locais de cada cano."""
        self.matriz_global = np.zeros((self.numero_nos, self.numero_nos))

        # Série das matrizes de cada cano
        for k, (idx_i, idx_j) in enumerate(self.conectividade):
            ck = self.condutancias[k]
            self.matriz_global[idx_i, idx_i] += ck
            self.matriz_global[idx_j, idx_j] += ck
            self.matriz_global[idx_i, idx_j] -= ck
            self.matriz_global[idx_j, idx_i] -= ck

    def resolver(self, nos_atm:list, bombas:dict):
        if self.matriz_global is None:
            self.assembly()
        
        """Resolve a rede utilizando análise nodal."""
        matriz_modificada = self.matriz_global.copy()
        vazao_modificada = np.zeroes(self.numero_nos)

        for no,vazao_bomba in bombas.items():
            index = no - 1
            vazao_modificada[index] +=vazao_bomba       
        
        # Nó de referência em que a pressão é zero.

        for no in nos_atm:
            index_atm = no - 1
            matriz_modificada[index_atm, :] = 0
            matriz_modificada[index_atm, index_atm] = 1
            vazao_modificada[index_atm] = 0
        self.pressao = np.linalg.solve(matriz_modificada,vazao_modificada)
        self.calcular_vazoes()

        self.historico_pressao.append(self.pressao.copy())
        self.historico_vazao.append(self.vazao.copy())

        return self.pressao

    def calcular_vazoes(self):
        """Calcula as vazões nos canos usando a fórmula Q = K * D * p."""
        # Matriz de incidência D
        numero_canos = len(self.conectividade)
        matriz_incidencia = np.zeros((numero_canos, self.numero_nos))
        for k, (i, j) in enumerate(self.conectividade):
            matriz_incidencia[k, i] = 1
            matriz_incidencia[k, j] = -1
        
        # Matriz diagonal de condutâncias K
        matriz_condutancias = np.diag(self.condutancias)

        # Computando a vazão com a expressão q = KDp 
        self.vazao = matriz_condutancias @ matriz_incidencia @ self.pressao
        
        return self.vazao

    def plotaRede(self, save_path=None):
        """Abre uma janela do matplotlib que plota o grafo."""
        if self.pressao is None or self.vazao is None:
            print("Erro: Resolva a rede antes de plotar.")
            return

        coord = self.posicoes_nos
        if coord is None:
            print("Erro: Coordenadas dos nós não fornecidas no construtor.")
            return

        edges = self.conectividade
        p = self.pressao
        q = self.vazao
        nv = self.numero_nos

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

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

def calcular_potencia_bomba(q_bomba,no_bomba,rede,p_noatm):
    pbomba = rede.p[no_bomba - 1]
    return (pbomba-p_noatm)*q_bomba

def calcular_potencias_bombas(bombas:dict, rede:RedeHidraulica, cenario_index:int = -1, p_noatm:float = 0.0):
    if not rede.historico_pressao:
        raise ValueError("A rede precisa estar resolvida para calcular a potência das bombas")
    pressoes_cenario = rede.historico_pressao[cenario_index]

    potencia_total = 0.0
    potencias_individuais = {}

    for no_bomba, q_bomba in bombas.items():
        index_bomba = no_bomba - 1
        p_bomba = pressoes_cenario[index_bomba]

        potencia = (p_bomba - p_noatm) * q_bomba

        potencias_individuais[no_bomba] = potencia
        potencia_total += potencia
        
    return potencia_total, potencias_individuais



