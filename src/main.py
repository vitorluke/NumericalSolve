from src.classes.rede_hidraulica import RedeHidraulica
from src.graphs_utils.utils import gerar_grafo_aleatorio, plotar_grafo_alternativo

def main():
# Definindo a geometria para o plot (x, y)
    coords = [
        [0, 0], # Nó 1
        [1, 0], # Nó 2
        [2, 0], # Nó 3
        [2, 1]  # Nó 4
    ]
    conec = [(1, 2), (2, 3), (3, 4)]
    conds = [10, 20, 30]
    
    rede = RedeHidraulica(n_nos=4, conectividade=conec, condutancias=conds, coordenadas=coords)
    #rede = gerar_grafo_aleatorio(4,3)
    rede.assembly()
    rede.resolver(no_atm=1, no_bomba=4, q_bomba=5)
    plotar_grafo_alternativo(rede)


if __name__ == "__main__":
    main()