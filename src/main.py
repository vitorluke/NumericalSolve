from src.classes.rede_hidraulica import RedeHidraulica
from src.graphs_utils.utils import gerar_grafo_aleatorio, plotar_grafo_alternativo
from src.graphs_utils.utils import gera_rede

def main():
    rede = gera_rede(3)

    rede.assembly()
    rede.resolver(no_atm=1, no_bomba=4, vazao_bomba=5)
    rede.plotaRede(1e-3)
    
    plotar_grafo_alternativo(rede)

if __name__ == "__main__":
    main()
