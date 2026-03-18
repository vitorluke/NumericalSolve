import math as math
from src.classes.rede_hidraulica import RedeHidraulica
from src.graphs_utils.gera_grafo import gera_grafo
from pyvis.network import Network
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

from src.graphs_utils.gera_grafo import gera_grafo

def gerar_grafo_aleatorio(numero_nos:int, numero_conexoes:int) -> RedeHidraulica:
    if numero_conexoes < numero_nos - 1:
        raise ValueError("Numero de conexões tem que ser no mínimo igual ao número de nós (do contrário, algum nó ficará sem conexão.)")

    rng = np.random.default_rng()
    condutancias = rng.uniform(low=0.1,high=20.0,size=numero_conexoes)
    coordenadas = rng.integers(low=0, high = 10, size=(numero_nos,2))
    nos = np.arange(numero_nos)
    rng.shuffle(nos)
    conexoes = []

    for i in range(numero_nos - 1):
        conexoes.append((nos[i],nos[i+1]))
    
    restante = numero_conexoes - len(conexoes)
    
    for _ in range(restante):
        pair = rng.choice(numero_nos,size=2,replace=False)
        conexoes.append(pair)
    conexoes = np.array(conexoes)
    
    return RedeHidraulica(numero_nos,conexoes,condutancias,coordenadas)

def plotar_grafo_alternativo(grafo: RedeHidraulica) -> None:
    """
    Gera um plot interativo com repulsão gravitacional (force-directed) 
    para a rede hidráulica.
    """
    if getattr(grafo, 'pressao', None) is None or getattr(grafo, 'vazao', None) is None:
        print("A rede não está resolvida.")
        return
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150, spring_strength=0.05, damping=0.09)

    p = grafo.pressao
    norm = mcolors.Normalize(vmin=float(p.min()), vmax=float(p.max()))
    cmap = cm.get_cmap("coolwarm")
    
    for i in range(grafo.numero_nos):
        rgba = cmap(norm(p[i]))
        hex_color = mcolors.to_hex(rgba)
        
        node_label = f"Nó {i+1}\np={p[i]:.2f}"
        net.add_node(i, label=node_label, color=hex_color, title=f"Pressão exata: {p[i]:.4f}")

    for idx, (u, v) in enumerate(grafo.conectividade):
        flow = grafo.vazao[idx]
        
        if p[u] > p[v]:
            source, target = u, v
        else:
            source, target = v, u

        net.add_edge(
            int(source), 
            int(target), 
            title=f"Vazão exata: {abs(flow):.4f}", 
            label=f"q={abs(flow):.2f}", 
            color="#ffffff", 
            arrows="to"
        )
    arquivo_saida = "rede_interativa.html"
    net.show(arquivo_saida, notebook=False)
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -10000,
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.02,
          "damping": 0.5,
          "avoidOverlap": 1
        },
        "maxVelocity": 10,
        "minVelocity": 0.1,
        "solver": "barnesHut",
        "stabilization": {
          "enabled": true,
          "iterations": 1000
        }
      }
    }
    """)

def gera_rede(levels=3):
    # Constantes relevantes
    A_k = 2.5e-7
    mu = 1e-3
    D_k = math.sqrt(4*A_k/math.pi)
    chi_k = math.pi * D_k**4 / (128 * mu)
        
    coordenadas, conectividade = gera_grafo(levels)

    numero_nos = len(coordenadas)

    condutancias = []

    for conexao in conectividade:
        no_1 = coordenadas[conexao[0]]
        no_2 = coordenadas[conexao[1]]

        L_k = math.sqrt((no_1[0]-no_2[0])**2+(no_1[1]-no_2[1])**2)
        condutancia = chi_k / L_k

        condutancias.append(condutancia)

    condutancias = np.array(condutancias)

    rede = RedeHidraulica(numero_nos, conectividade, condutancias, coordenadas)

    return rede