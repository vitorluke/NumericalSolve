from src.classes.rede_hidraulica import RedeHidraulica
from src.graphs_utils.utils import gerar_grafo_aleatorio, plotar_grafo_alternativo
from src.graphs_utils.utils import gera_rede
from src.graph_benchmarking.benchmark import *
import src.classes.placa_termica as pt
from collections.abc import Mapping


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
        import numpy as np
from src.classes.rede_hidraulica import RedeHidraulica, calcular_potencias_bombas
from src.graphs_utils.utils import gerar_grafo_aleatorio, plotar_grafo_alternativo
from src.graphs_utils.utils import gera_rede
from src.graph_benchmarking.benchmark import *




def rede_hidraulica():
    str_input = ""
    pressoes_impostas = {5: 0.0, 10: 0.0, 15: 0.0, 20: 0.0}
    vazoes_impostas = {2: 0.25, 7: 0.25, 12: 0.25, 17: 0.25}
    type_of_generation = int(input("1 - Geração aleatória por número de nós\n2 - Geração aleatória por level\nEscolha: "))
    
    if type_of_generation == 1:
        str_input = "Número de nós: "
    else:
        str_input = "Número de levels: "
        
    amount = int(input(str_input))
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
        
        print("\n" + "="*55)
        print("OVERVIEW FINAL DA REDE")
        print("="*55)
        
        print("\nINFORMAÇÕES GERAIS")
        print(f"   • Número de Nós:   {rede.numero_nos}")
        print(f"   • Número de Tubos: {len(rede.conectividade)}")
        
        print("\nGRANDEZAS HIDRÁULICAS")
        print(f"   • Pressão Nodal (p)  -> Mín: {np.min(rede.pressao):.4f} | Máx: {np.max(rede.pressao):.4f} | Média: {np.mean(rede.pressao):.4f}")
        print(f"   • Vazão em Tubos (q) -> Absoluta Máx: {np.max(np.abs(rede.vazao)):.4f} | Média: {np.mean(np.abs(rede.vazao)):.4f}")
        
        print("\nBALANÇO DE ENERGIA (POTÊNCIA)")
        potencia_dissipada = rede.calcular_potencia()
        if potencia_dissipada is not None:
            p_diss = potencia_dissipada.item() if isinstance(potencia_dissipada, np.ndarray) else potencia_dissipada
            print(f"   • Potência Total Dissipada na Rede: {p_diss:.4f} W")
        
        if vazoes_impostas:
            pot_total_bombas, pot_individuais = calcular_potencias_bombas(vazoes_impostas, rede, p_noatm=0.0)
            print(f"   • Potência Total Fornecida pelas Bombas: {pot_total_bombas:.4f} W")
            for no, pot in pot_individuais.items():
                print(f"     - Bomba instalada no Nó {no}: {pot:.4f} W")
                
        print("\n" + "="*55 + "\n")
        
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



def print_options(function_map:Mapping[str,function]):
    map = {}
    print("Menu:")
    print("-"*20)
    for index,value in enumerate(function_map.keys()):
        map[index] = function_map[value]
        print(f"{index+1:02} - {value}")
    print("-"*20)
    return map

def function_menu(map:Mapping[str,function]):
    index_mapping = print_options(map)
    choice = int(input("Digite o número da opção: ")) - 1
    if index_mapping.__contains__(choice):
        return index_mapping[choice]()
    else:
        print("Opção inválida!")
        function_menu(map)




def main():
    function_menu({
        "Exercício 1": pt.exercicio_1,
        "Exercício 2": pt.exercicio_2,
        "Exercício 3": pt.exercicio_3,
        "Exercício 4": pt.exercicio_4,
        "Exercício 5": pt.exercicio_5,
        "Exercício 1 Extra": pt.exercicio_1_extra,
        "Exercício 2 Extra": pt.exercicio_2_extra,
        "Exercício 3 Extra": pt.exercicio_3_extra,
    })

    

if __name__ == "__main__":
    main()
