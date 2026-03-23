from src.classes.rede_hidraulica import RedeHidraulica
from src.graphs_utils.utils import gerar_grafo_aleatorio, plotar_grafo_alternativo
from src.graphs_utils.utils import gera_rede
from src.graph_benchmarking.benchmark import *


def main():
    str = ""
    pressoes_impostas = {2: 0.0}
    vazoes_impostas = {1: 0.1}
    type_of_generation = int(input("1 - Geração aleatória por número de nós\n2 - Geração aleatória por level\nEscolha: "))
    if type_of_generation == 1:
        str = "Número de nós: "
    else:
        str = "Número de levels: "
    amount = int(input(str))
    rede = gera_rede(0)
    nos_atm = rede.numero_nos
    if type_of_generation == 1:
        rede = gerar_grafo_aleatorio(amount, amount + 4)
    else:
        rede = gera_rede(amount)
    n = int(input("1 - Resolução de Rede\n2 - Benchmarks\nEscolha: "))

    niveis = [2,3,4,5]

    if n == 1:
        rede.assembly()
        rede.resolver(pressao_imposta=pressoes_impostas, vazao_imposta=vazoes_impostas)
        plotting_type = int(input("1 - Plot padrão\n2 - Plot alternativo\nEscolha: "))
        if plotting_type == 2:
            plotar_grafo_alternativo(rede)
        else:
            rede.plotaRede(1e3)
    else:
        exercise_num = int(input("1 - Exercício 4\n2 - Exercício 5\n3 - Exercício 6\n4 - Exercício 7\nEscolha: "))
        match exercise_num:
            case 1:
                exercicio_4(rede,pressoes_impostas)
            case 2:
                exercicio_5(rede,pressoes_impostas)
            case 3:
                exercicio_6(rede,pressoes_impostas)
            case 4:
                exercicio_7(niveis)
            case _:
                print("Escolha não existe!")

    rede = gera_rede(1)
    rede.assembly()
    pressao = rede.resolver(pressao_imposta={6: 0.0}, vazao_imposta={1: 0.1})
    #rede.plotaRede(1e3)
    

    #print("Executando Exercícios 4, 5 e 6...")
    #rede_grande = gera_rede(3) 
    #ede_grande.assembly()
    #nos_atm = [rede_grande.numero_nos]

    #exercicio_4(rede_grande, nos_atm)
    #exercicio_5(rede_grande, nos_atm)
    #exercicio_6(rede_grande, nos_atm)

    #niveis_para_testar = [1,2,3,4,5,6,7] 
    #exercicio_7(niveis_para_testar)
    
    #print("--- Processo Finalizado ---")

if __name__ == "__main__":
    main()
