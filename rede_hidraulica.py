import __main__
import numpy as np

class RedeHidraulica:
    def __init__(self, n_nos, conectividade, condutancias):
        self.n_nos = n_nos
        self.conec = np.array(conectividade)
        self.C = np.array(condutancias)
        self.A = np.zeros((n_nos, n_nos))
        self.p = None

    def assembly(self):
        """Monta a matriz global A a partir das matrizes locais."""
        for k, (i, j) in enumerate(self.conec):
            idx_i, idx_j = i-1, j-1 
            ck = self.C[k]
            
            # Matriz local contribuindo para a global
            self.A[idx_i, idx_i] += ck
            self.A[idx_j, idx_j] += ck
            self.A[idx_i, idx_j] -= ck
            self.A[idx_j, idx_i] -= ck

    def resolver(self, no_atm, no_bomba, q_bomba):
        """Aplica condições de contorno e resolve Ax = b."""
        Atilde = self.A.copy()
        b = np.zeros(self.n_nos)
        
        # Pressão fixada (ex: p_atm = 0)
        idx_atm = no_atm - 1
        Atilde[idx_atm, :] = 0
        Atilde[idx_atm, idx_atm] = 1
        
        # Vazão da bomba
        b[no_bomba - 1] = q_bomba
        
        self.p = np.linalg.solve(Atilde, b)
        return self.p

    def calcular_potencia(self):
        # A FAZER: Implementar cálculo de potência hidráulica
        pass

if __main__.__name__ == "__main__":
    # Exemplo de uso, seja feliz!
    # C.I: Pressão no nó 1 é atmosférica (0), bomba no nó 4 com vazão de 5 unidades.
    n_nos = 4
    conectividade = [(1, 2), (2, 3), (3, 4)]
    condutancias = [10, 20, 30]
    no_atm = 1
    no_bomba = 4
    q_bomba = 5
    
    rede = RedeHidraulica(n_nos, conectividade, condutancias)
    rede.assembly()
    
    
    
    pressao = rede.resolver(no_atm, no_bomba, q_bomba)
    print("Pressões nos nós:", pressao)