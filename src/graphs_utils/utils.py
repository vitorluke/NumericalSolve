from src.classes.rede_hidraulica import RedeHidraulica
import numpy as np

def gerar_grafo_aleatorio(numero_nos:int, numero_conexoes:int) -> RedeHidraulica:
    rng = np.random.default_rng()
    condutancias = rng.uniform(low=0.1,high=20.0,size=numero_conexoes)
    coordenadas = rng.integers(low=0, high = 10, size=(numero_nos,2))
    conexoes = rng.integers(low=0,high=numero_nos, size=(numero_conexoes,2))
    return RedeHidraulica(numero_nos,conexoes,condutancias,coordenadas)

def plotar_grafo_alternativo(grafo:RedeHidraulica) -> None:
    return