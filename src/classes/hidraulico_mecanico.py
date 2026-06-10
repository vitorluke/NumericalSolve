import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from src.classes.membrana_elastica import MembranaElastica
from src.classes.rede_hidraulica import RedeHidraulica

class HidraulicoMecanico:
    def __init__(self, N_mem=51, H_k=1000e-6, beta_hat=0.1, n_levels=3):
        """
        Inicializa o modelo acoplado Hidráulico-Mecânico.
        """
        # Parâmetros Físicos Nominais
        self.N_mem = N_mem
        self.R_dim = 0.25e-2
        self.e_dim = 0.1e-3
        self.sigma = 200.0
        self.rho = 900.0
        self.mu = 5e-4
        self.H_k = H_k
        self.beta_hat = beta_hat
        
        # Índices Críticos da Rede
        self.nin = 0
        self.nout = 5
        
        # Inicialização das Componentes Isoladas
        self.membrana = MembranaElastica(N=self.N_mem, R=self.R_dim)
        self.rede = RedeHidraulica(levels=n_levels)
        
        self.nm = self.membrana.nunk
        self.np_nodes = self.rede.numero_nos
        
        # Escalas de Adimensionalização
        self.w0 = 0.01 * self.R_dim
        self.pref = (self.sigma * self.w0) / (self.R_dim**2)
        self.vref = np.sqrt(self.sigma / (self.rho * self.e_dim)) * (self.w0 / self.R_dim)
        self.t_ref = self.R_dim * np.sqrt((self.rho * self.e_dim) / self.sigma)
        
        self.h_hat = 2.0 / (self.N_mem - 1)
        
        self._preparar_matrizes()

    def _preparar_matrizes(self):
        """
        Ajusta propriedades, aplica condições de contorno e adimensionaliza os sistemas.
        """
        # 1. Recalcular Condutâncias da Rede para seção quadrada e H_k especificado
        kappa_k = (np.pi * (self.H_k**4)) / (128 * self.mu)
        condutancias = []
        for (i, j) in self.rede.conectividade:
            n1 = self.rede.posicoes_nos[i]
            n2 = self.rede.posicoes_nos[j]
            L_k = np.sqrt((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2)
            condutancias.append(kappa_k / L_k)
        
        self.rede.condutancias = np.array(condutancias)
        self.rede.assembly()
        
        # A dimensional (usada puramente para cálculo de potência mecânica)
        self.A_dim_pura = self.rede.matriz_global.copy()
        
        # Adimensionalização da Matriz A
        A_scale = self.pref / (self.vref * (self.R_dim**2))
        A_adim_array = self.A_dim_pura * A_scale
        
        # Condição de Dirichlet na entrada (Nó 0)
        A_adim_array[self.nin, :] = 0.0
        A_adim_array[self.nin, self.nin] = 1.0
        self.A_adim = sp.csr_matrix(A_adim_array)
        
        # 2. Matrizes da Membrana
        self.K = self.membrana.K
        self.M = self.membrana.M
        self.D = self.beta_hat * self.M
        
        # 3. Matriz de Acoplamento U
        U = np.zeros((self.np_nodes, self.nm))
        self.uns = np.ones(self.nm)
        
        for i in range(self.N_mem):
            for j in range(self.N_mem):
                idx = i + j * self.N_mem
                x = i * self.h_hat - 1.0
                y = j * self.h_hat - 1.0
                dist_sq = x**2 + y**2
                
                # Zera nós fora do contorno circular e nas bordas rígidas quadradas
                if dist_sq > 1.0 or i == 0 or i == self.N_mem-1 or j == 0 or j == self.N_mem-1:
                    self.uns[idx] = 0.0
                    
        U[self.nout, :] = self.uns
        self.U = sp.csr_matrix(U)

    def solver_transiente(self, dt_hat, t_final_hat, p_inlet_func, estado_inicial=None, idx_monitor=None):
        """
        Resolvedor dinâmico utilizando Euler Implícito em blocos.
        """
        n_steps = int(t_final_hat / dt_hat)
        idt = 1.0 / dt_hat
        Iden = sp.identity(self.nm, format='csr')
        
        # Montagem do Aglob
        blocks = [
            [idt * Iden, -Iden, None],
            [self.K, idt * self.M + self.D, -self.U.T],
            [None, self.U * (self.h_hat**2), self.A_adim]
        ]
        Aglob = sp.bmat(blocks, format='csc')
        solver_LU = spla.factorized(Aglob)
        
        if estado_inicial is None:
            w = np.zeros(self.nm)
            v = np.zeros(self.nm)
            p = np.zeros(self.np_nodes)
            t_hat = 0.0
        else:
            w, v, p, t_hat = estado_inicial
            
        hist = {'t': [], 'w_center': [], 'p_out': [], 'q_out': [], 'volume': [], 'potencia': [], 'w_full': None}
        
        # SE NENHUM NÓ FOR ESPECIFICADO, MONITORA O CENTRO POR PADRÃO
        if idx_monitor is None:
            idx_monitor = (self.N_mem // 2) + (self.N_mem // 2) * self.N_mem
        
        for _ in range(n_steps):
            t_hat += dt_hat
            p_inlet_dim = p_inlet_func(t_hat)
            p_inlet_hat = p_inlet_dim / self.pref
            
            b_mec = idt * w
            b_vel = idt * self.M.dot(v)
            b_hid = np.zeros(self.np_nodes)
            b_hid[self.nin] = p_inlet_hat
            
            b_global = np.concatenate([b_mec, b_vel, b_hid])
            sol = solver_LU(b_global)
            
            w, v, p = sol[0:self.nm], sol[self.nm:2*self.nm], sol[2*self.nm:]
            
            # Reversão Dimensional para Histórico
            p_out_dim = p[self.nout] * self.pref
            q_out_dim = (self.h_hat**2) * np.sum(v * self.uns) * (self.vref * self.R_dim**2)
            w_cen_dim = w[idx_monitor] * self.w0  # MONITORANDO O NÓ DE INTERESSE
            vol_dim = (self.h_hat**2) * np.sum(w * self.uns) * (self.w0 * self.R_dim**2)
            pot_dim = (p.T @ self.A_dim_pura @ p) * (self.pref * self.vref * self.R_dim**2)
            
            hist['t'].append(t_hat)
            hist['p_out'].append(p_out_dim)
            hist['q_out'].append(q_out_dim)
            hist['w_center'].append(w_cen_dim)
            hist['volume'].append(vol_dim)
            hist['potencia'].append(pot_dim)
            
        hist['w_full'] = w * self.w0
        return hist, (w, v, p, t_hat)


    # ==========================================
    # MÉTODOS DOS EXERCÍCIOS ESPECÍFICOS
    # ==========================================

    @classmethod
    def executar_ex_01(cls):
        print("=== Exercício 1: Estrutura da Matriz R ===")
        # Malha reduzida solicitada
        sistema = cls(N_mem=26)
        
        A_inv = np.linalg.inv(sistema.A_adim.toarray())
        U_denso = sistema.U.toarray()
        
        R = (sistema.h_hat**2) * (U_denso.T @ A_inv @ U_denso)
        
        plt.figure(figsize=(6, 6))
        plt.spy(R, marker=',', color='blue', precision=1e-10)
        plt.title('Estrutura de Esparsidade da Matriz $R$ ($26 \\times 26$)')
        plt.xlabel('Graus de Liberdade da Membrana')
        plt.ylabel('Graus de Liberdade da Membrana')
        plt.tight_layout()
        plt.show()

    @classmethod
    def executar_ex_02(cls):
        print("=== Exercício 2: Enchimento/Pressurização ===")
        sistema = cls(N_mem=51, H_k=1000e-6, beta_hat=0.1)
        
        # Pressão Degrau de 10kPa
        def p_in(t): return 10000.0
        
        hist, estado_final = sistema.solver_transiente(dt_hat=0.025, t_final_hat=12.0, p_inlet_func=p_in)
        cls._plotar_6_graficos(hist, "Exercício 2: Resposta ao Degrau de Pressão (10 kPa)")
        
        return estado_final

    @classmethod
    def executar_ex_03(cls, estado_inicial_ex2):
        print("=== Exercício 3: Relaxação Instântanea ===")
        sistema = cls(N_mem=51, H_k=1000e-6, beta_hat=0.1)
        
        # Pressão Cai para Zero
        def p_in(t): return 0.0
        
        hist, _ = sistema.solver_transiente(dt_hat=0.025, t_final_hat=12.0, p_inlet_func=p_in, estado_inicial=estado_inicial_ex2)
        cls._plotar_6_graficos(hist, "Exercício 3: Relaxação (Pressão Inlet = 0)")

    @classmethod
    def executar_ex_04(cls, modo=3):
        index_modo = modo-1
        print(f"=== Exercício 4: Oscilação Livre do {modo}º Modo ===")
        # Sem amortecimento intrínseco e canais largos
        sistema = cls(N_mem=51, H_k=2000e-6, beta_hat=0.0)
        
        # Obter o 3º Modo
        freqs, omegas, modos = sistema.membrana.solve_modes_adimensional(nmodes=10)
        w3_hat = omegas[index_modo] 
        modo_3 = modos[:, index_modo]
        
        print(f"Frequência Isolada da Membrana (Adimensional w3): {w3_hat:.4f}")
        
        # Condição inicial customizada (amplitude máxima de 0.1 adimensional)
        w_init = (modo_3 / np.max(np.abs(modo_3))) * 0.1
        estado_zero = (w_init, np.zeros(sistema.nm), np.zeros(sistema.np_nodes), 0.0)
        
        def p_in(t): return 0.0
        
        # ENCONTRA O NÓ DE DEFLEXÃO MÁXIMA PARA CORRIGIR A ESCALA DO GRÁFICO
        idx_max = np.argmax(np.abs(w_init))
        
        # Passando o idx_max para o solver monitorar
        hist, _ = sistema.solver_transiente(dt_hat=0.01, t_final_hat=30.0, p_inlet_func=p_in, 
                                            estado_inicial=estado_zero, idx_monitor=idx_max)
        
        # Encontrar frequência observada via cruzamento por zero
        w_c = np.array(hist['w_center'])
        zeros = np.where(np.diff(np.sign(w_c - np.mean(w_c))))[0]
        if len(zeros) >= 2:
            periodo = 2 * np.mean(np.diff(np.array(hist['t'])[zeros]))
            freq_obs = (2 * np.pi) / periodo
            print(f"Frequência Acoplada Observada (Adimensional): {freq_obs:.4f}")
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(hist['t'], hist['w_center'], color='green')
        ax[0].set(title="Ex 4: Deflexão no Ponto de Maior Amplitude", xlabel="Tempo $\hat{t}$", ylabel="$w$ (m)")
        ax[0].grid(True)
        
        ax[1].plot(hist['t'], hist['p_out'], color='darkgreen')
        ax[1].set(title="Ex 4: Pressão de Saída (Balanço de Volume)", xlabel="Tempo $\hat{t}$", ylabel="$p$ (Pa)")
        ax[1].grid(True)
        plt.tight_layout()
        plt.show()
        
        return w3_hat

    @classmethod
    def executar_ex_05(cls, w3_hat):
        print("=== Exercício 5: Ressonância Harmônica ===")
        sistema = cls(N_mem=51, H_k=2000e-6, beta_hat=0.0)
        
        # Excitação Harmônica na Frequência Natural
        def p_in(t): return 5000.0 * np.cos(w3_hat * t)
        
        hist, _ = sistema.solver_transiente(dt_hat=0.01, t_final_hat=30.0, p_inlet_func=p_in)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(hist['t'], hist['w_center'], color='blue')
        ax[0].set(title="Ex 5: Ressonância (Deflexão)", xlabel="Tempo $\hat{t}$", ylabel="$w$ (m)")
        ax[0].grid(True)
        
        ax[1].plot(hist['t'], hist['p_out'], color='red')
        ax[1].set(title="Ex 5: Pressão Transiente", xlabel="Tempo $\hat{t}$", ylabel="$p$ (Pa)")
        ax[1].grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plotar_6_graficos(hist, titulo_principal):
        fig, axs = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle(titulo_principal, fontsize=14, fontweight='bold')
        
        axs[0, 0].plot(hist['t'], hist['p_out'], color='blue')
        axs[0, 0].set(title='$p_{outlet}$', xlabel='$\hat{t}$', ylabel='Pa')
        axs[0, 0].grid(True)
        
        axs[0, 1].plot(hist['t'], hist['q_out'], color='orange')
        axs[0, 1].set(title='$q_{outlet}$ (Vazão)', xlabel='$\hat{t}$', ylabel='$m^3/s$')
        axs[0, 1].grid(True)
        
        axs[0, 2].plot(hist['t'], hist['w_center'], color='green')
        axs[0, 2].set(title='Deflexão Central', xlabel='$\hat{t}$', ylabel='m')
        axs[0, 2].grid(True)
        
        axs[1, 0].plot(hist['t'], hist['volume'], color='purple')
        axs[1, 0].set(title='Volume Reservatório', xlabel='$\hat{t}$', ylabel='$m^3$')
        axs[1, 0].grid(True)
        
        axs[1, 1].plot(hist['t'], hist['potencia'], color='red')
        axs[1, 1].set(title='Potência Consumida', xlabel='$\hat{t}$', ylabel='Watts')
        axs[1, 1].grid(True)
        
        X_m, Y_m = np.meshgrid(np.linspace(0, 0.5, int(np.sqrt(len(hist['w_full'])))), 
                               np.linspace(0, 0.5, int(np.sqrt(len(hist['w_full'])))))
        im = axs[1, 2].contourf(X_m, Y_m, hist['w_full'].reshape(X_m.shape), 20, cmap='viridis')
        fig.colorbar(im, ax=axs[1, 2], label='m')
        axs[1, 2].set(title='Perfil Final da Membrana')
        axs[1, 2].set_aspect('equal')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Execução sequencial encapsulada
    #HidraulicoMecanico.executar_ex_01()
    #estado_ex2 = HidraulicoMecanico.executar_ex_02()
    #HidraulicoMecanico.executar_ex_03(estado_ex2)
    w3_hat = HidraulicoMecanico.executar_ex_04(modo=1)
    #HidraulicoMecanico.executar_ex_05(w3_hat)