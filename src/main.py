from src.classes.rede_hidraulica import RedeHidraulica
from src.graphs_utils.utils import gerar_grafo_aleatorio, plotar_grafo_alternativo
from src.graphs_utils.utils import gera_rede
from src.graph_benchmarking.benchmark import *


def main():
    rede = gera_rede(3)

    rede.assembly()
    #rede.resolver(no_atm=1, no_bomba=4, vazao_bomba=5)
    #rede.plotaRede(1e-3)
    print("Executando Exercícios 4, 5 e 6...")
    rede_grande = gera_rede(3) 
    rede_grande.assembly()
    nos_atm = [rede_grande.numero_nos]

    #exercicio_4(rede_grande, nos_atm)
    #exercicio_5(rede_grande, nos_atm)
    #exercicio_6(rede_grande, nos_atm)

    niveis_para_testar = [1,2,3,4,5,6,7] 
    exercicio_7(niveis_para_testar)
    
    print("--- Processo Finalizado ---")
    
    plotar_grafo_alternativo(rede)

if __name__ == "__main__":
    main()
