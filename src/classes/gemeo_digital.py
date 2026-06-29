import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time
import warnings
from scipy.interpolate import interp1d

# Ocultar avisos transientes e de otimização
warnings.simplefilter('ignore')

from src.classes.rede_hidraulica import RedeHidraulica
from src.classes.placa_termica import PlacaTermica
from src.classes.membrana_elastica import MembranaElastica

from src.classes.hidraulico_termico import HidraulicoTermico
from src.classes.hidraulico_mecanico import HidraulicoMecanico

class GemeoDigital:
    def __init__(self, levels_rede=3):
        """Inicializa o Gêmeo Digital instanciando as três físicas principais."""
        print("[GD] Inicializando os subsistemas físicos...")
        self.H_k = 1e-3

        self.rede = RedeHidraulica(levels=levels_rede, H_k=self.H_k)
        self.placa = PlacaTermica(Lx=0.03, Ly=0.015, Nx=61, Ny=31, k=0.25, R=0.0025, fonte_calor=5e5)
        self.membrana = MembranaElastica(N=51, R=0.0025)
        
        self.acop_hidrotermico  =  HidraulicoTermico(self.rede, self.placa)
        self.acop_hidromecanico = HidraulicoMecanico(self.rede, self.membrana)

        self.p_inlet = 5e3

        # Estruturas para guardar o histórico temporal (Usado no ex_4)
        self.hist_mono = None
        self.hist_part = None
        
        self.preparar_estado_nominal()

    def preparar_estado_nominal(self):
        """Define e guarda as condutâncias nominais a 20ºC para usar como baseline."""
        T_nominal = np.full(self.rede.numero_nos, 20.0)
        self.rede.atualizar_condutancias(T_nominal)
        self.rede.assembly()
        self.condutancias_originais = self.rede.condutancias.copy()
        
        self.vazao_nominal = self._extrair_vazao()
        # Limite crítico exato da seção 6.3.2 da apostila do curso
        self.limite_critico = 1.25e-5
        print(f"[GD] Vazão Nominal (20°C): {self.vazao_nominal:.4e} m³/s")
        print(f"[GD] Limite Crítico de Projeto: {self.limite_critico:.4e} m³/s")

    # =========================================================================
    # EX 1.1: Análise estacionária de falhas hidraulicas
    # =========================================================================
        # Ex 1.1 i RandomFail()
    def _RandomFail(self, p_O, f_obs):
        C_modificado = self.condutancias_originais.copy()
        C_modificado[np.random.rand(len(C_modificado)) < p_O] /= f_obs

        return C_modificado
    
    def _extrair_vazao(self):
        try:
            p_sol = self.rede.resolver(pressao_imposta={1: self.p_inlet, 6: 0.0})

            row = self.rede.matriz_global.getrow(0)

            q = row @ p_sol

            return float(np.asarray(q).squeeze())

        except np.linalg.LinAlgError:
            return 0.0
    
    def ex_1_1(self):
        print("\n" + "="*60)
        print("EX 1.1: Análise estacionária de falhas hidraulicas")
        print("="*60)

        # 1.1 ii  Convergência Monte Carlo
        
        N_amostras = 5000
        print(f"\n[EX 1.1 i] Convergência Monte Carlo (N={N_amostras})...")
        falhas_criticas = 0
        Y = []

        for i in range(1, N_amostras + 1):
            self.rede.condutancias = self._RandomFail(p_O=0.35, f_obs=5.0)
            self.rede.assembly()
            vazao = self._extrair_vazao()
            if vazao < self.limite_critico:
                falhas_criticas += 1
            
            Y.append(falhas_criticas/i*100.0)

        Prob = falhas_criticas/N_amostras*100
        X = np.linspace(1, N_amostras, N_amostras)
        print(f"-> Probabilidade Final Estabilizada: {Prob:.2f}%")

        plt.figure(figsize=(9,5))
        plt.plot(X, Y)
        plt.axhline(Prob, linestyle='--', label=f"Assíntota $\\approx {Prob:.2f}\\%$")
        plt.ylim(0, 100)
        plt.title("Convergência do método de Monte Carlo")
        plt.xlabel("Iterações")
        plt.ylabel("Probabilidade global de falha (%)")
        plt.legend()
        plt.savefig("imagens/gêmeo digital/ex 1_1/ii.png")
        plt.show()

        # 1.1 iii Domínio de probabilidade
        
        p_O_lin = np.linspace(0.05, 0.65, 7)
        Prob = {5: [], 10: []}

        for p_O in p_O_lin:
            for f_obs in Prob:
                falhas_criticas = 0
                for i in range(1, N_amostras + 1):
                    self.rede.condutancias = self._RandomFail(p_O=p_O, f_obs=f_obs)
                    self.rede.assembly()
                    vazao = self._extrair_vazao()

                    if vazao < self.limite_critico:
                        falhas_criticas += 1
            
                Prob[f_obs].append(falhas_criticas/N_amostras*100.0)

        plt.figure(figsize=(9,5))
        plt.plot(p_O_lin, Prob[5], '-o', label='$f_{obs}=5$')
        plt.plot(p_O_lin, Prob[10], '-o', label='$f_{obs}=10$')
        plt.ylim(0, 100)
        plt.title("Domínio de probabilidade")
        plt.xlabel("$p_O$")
        plt.ylabel("Probabilidade global de falha (%)")
        plt.legend()
        plt.savefig("imagens/gêmeo digital/ex 1_1/iii.png")
        plt.show()

    # =========================================================================
    # EX 1.2 Análise Dinâmica do Gêmeo Digital Completo
    # =========================================================================
    def solver_transiente(self, dt_hat, t_final_hat):

        n_steps = int(t_final_hat / dt_hat)
        t_hat = 0.0

        acop = self.acop_hidromecanico

        nm = acop.nm
        np_nodes = acop.np_nodes
        nin = acop.nin

        w = np.zeros(nm)
        v = np.zeros(nm)
        p = np.zeros(np_nodes)

        N = acop.N_mem
        idx_monitor = (N // 2) + (N // 2) * N

        dt = dt_hat
        idt = 1.0 / dt

        pref = acop.pref
        vref = acop.vref
        R_dim = acop.R_dim
        w0 = acop.w0

        h2 = acop.h_hat ** 2
        q_scale = vref * R_dim**2
        power_scale = (pref**2) * 2.0 / R_dim

        M = acop.M
        uns = acop.uns

        # cache funções (ganho leve mas real em loop Python)
        temperaturas_medias_arestas = self.acop_hidrotermico.temperaturas_medias_arestas
        calcular_viscosidade = self.acop_hidrotermico.calcular_viscosidade

        I = sp.identity(nm, format='csr')

        blocks = [
            [idt * I, -I, None],
            [acop.K, idt * M + acop.D, -acop.U.T],
            [None, acop.U * h2, acop.A_adim]
        ]

        Aglob = sp.bmat(blocks, format='csc')
        solver = spla.factorized(Aglob)

        b_global = np.zeros(2 * nm + np_nodes)

        hist = {
            't': np.empty(n_steps),
            'w_center': np.empty(n_steps),
            'p_out': np.empty(n_steps),
            'q_out': np.empty(n_steps),
            'power': np.empty(n_steps),
            'energy': np.empty(n_steps)
        }

        energy = 0.0

        w_slice = slice(0, nm)
        v_slice = slice(nm, 2 * nm)
        p_slice = slice(2 * nm, 2 * nm + np_nodes)

        nin_idx = 2 * nm + nin

        for k in range(n_steps):

            # ----------------------------
            # atualização térmica
            # ----------------------------
            T_med, _ = temperaturas_medias_arestas()
            mu_med = calcular_viscosidade(T_med)

            # rede ainda pode atualizar, mas NÃO entra mais na potência
            self.rede.atualizar_condutancias(mu_med)

            # ----------------------------
            # RHS (sem overhead)
            # ----------------------------
            b_global[w_slice] = idt * w
            b_global[v_slice] = idt * (M @ v)
            b_global[p_slice] = 0.0
            b_global[nin_idx] = self.p_inlet / pref

            sol = solver(b_global)

            w = sol[w_slice]
            v = sol[v_slice]
            p = sol[p_slice]

            # ----------------------------
            # métricas leves
            # ----------------------------
            p_out_dim = p[5] * pref

            q_out_dim = h2 * (v @ uns) * q_scale

            w_cen_dim = w[idx_monitor] * w0

            # potência AGORA local (sem matriz global)
            # substitui G @ p completamente
            # usa interpretação de dissipação hidráulica simples:
            P_inst = power_scale * np.sum((p[acop.nin] - p) ** 2)

            energy += P_inst * dt

            # ----------------------------
            # histórico
            # ----------------------------
            hist['t'][k] = t_hat
            hist['w_center'][k] = w_cen_dim
            hist['p_out'][k] = p_out_dim
            hist['q_out'][k] = q_out_dim
            hist['power'][k] = P_inst
            hist['energy'][k] = energy

            t_hat += dt

        return hist, (w, v, p, t_hat, energy)
    
    def ex_1_2(self, p_O=0.20, f_obs=5.0):
        N_amostras = 2000
        dt_values = [0.05, 0.1]

        for dt in dt_values:
            print(f"[dt={dt}]")
            X = np.linspace(1, N_amostras, N_amostras)
            Y = []
            num = 0
            for i in range(1, N_amostras + 1):
                self.rede.condutancias = self._RandomFail(p_O=p_O, f_obs=f_obs)
                _, (_, _, _, _, energy) = self.solver_transiente(dt, 4.0)

                if energy < 7:
                    num += 1

                if i % 100 == 0:
                    print(f"{i}/{N_amostras}...")

                Y.append(num / N_amostras * 100.0)

            Prob = num / N_amostras * 100.0
            print(f"-> Probabilidade Final Estabilizada: {Prob:.2f}%")

            plt.figure(figsize=(9,5))
            plt.plot(X, Y)
            plt.axhline(Prob, linestyle='--', label=f"Assíntota $\\approx {Prob:.2f}\\%$")
            plt.ylim(0, 100)
            plt.title("Convergência do método de Monte Carlo")
            plt.xlabel("Iterações")
            plt.ylabel("Probabilidade global $E<7.0$ (%)")
            plt.legend()
            plt.savefig(f"imagens/gêmeo digital/ex 1_2/{dt}.png")
            plt.show()

    # =========================================================================
    # EX 1 ESPECIAL: CAMPO TÉRMICO COM MICRO REATOR A 45°C
    # =========================================================================
    def ex_1_especial(self):
        print("\n" + "="*60)
        print("EX 1 ESPECIAL: CAMPO TÉRMICO COM MICRO REATOR A 45°C")
        print("="*60)
        self.placa.R = 0.0025
        print("[GD] Resolvendo a Placa Térmica (Tc = 45°C)...")
        self.placa.resolver_circulo(Tc=45.0, mode='sparse')
        print(f"  -> Temperatura Máxima: {self.placa.temp_max():.2f} °C")
        print(f"  -> Temperatura Média:  {self.placa.temp_med():.2f} °C")
        self.placa.plota_placa(flag_type='contour', title="Ex 1 Especial: Placa Térmica (Tc=45°C, R=2.5mm)")

    # =========================================================================
    # EX 1: ACOPLAMENTO TÉRMICO-HIDRÁULICO (NOMINAL)
    # =========================================================================
    def ex_1(self):
        print("\n" + "="*60)
        print("EX 1: ACOPLAMENTO TÉRMICO-HIDRÁULICO (NOMINAL)")
        print("="*60)
        T_nominal = np.full(self.rede.numero_nos, 20.0)
        self.rede.atualizar_condutancias(T_nominal)
        self.rede.assembly()
        self.rede.resolver(pressao_imposta={1: 5000.0}, vazao_imposta={175: 0.0})
        W_20 = float(self.rede.calcular_potencia())
        
        self.placa.resolver_borda(mode='sparse')
        pos_x, pos_y = self.rede.posicoes_nos[:, 0], self.rede.posicoes_nos[:, 1]
        T_nos_rede = self.placa.temperatura_em(np.column_stack((pos_x, pos_y)))

        self.rede.atualizar_condutancias(T_nos_rede)
        self.rede.assembly()
        self.rede.resolver(pressao_imposta={1: 5000.0}, vazao_imposta={175: 0.0})
        W_acoplado = float(self.rede.calcular_potencia())
        
        print(f"  -> Potência Isotérmico 20°C: {W_20:.6e} W")
        print(f"  -> Potência c/ Acoplamento:  {W_acoplado:.6e} W")
        print(f"[Conclusão] Variação na potência: {((W_acoplado - W_20) / W_20) * 100:+.2f}%\n")
        self.rede.condutancias = self.condutancias_originais.copy()

    # =========================================================================
    # EX 2 ESPECIAL: IMPACTO DO RAIO DO REATOR NA POTÊNCIA (VARREDURA)
    # =========================================================================
    def ex_2_especial(self):
        print("\n" + "="*60)
        print("EX 2 ESPECIAL: IMPACTO DO RAIO DO REATOR NA POTÊNCIA")
        print("="*60)
        raios_mm = np.arange(1.0, 5.5, 0.5)
        raios_m = raios_mm / 1000.0
        potencias = []
        pos_x, pos_y = self.rede.posicoes_nos[:, 0], self.rede.posicoes_nos[:, 1]
        pts_rede = np.column_stack((pos_x, pos_y))
        
        for r_mm, r_m in zip(raios_mm, raios_m):
            self.placa.R = r_m
            self.placa.resolver_circulo(Tc=45.0, mode='sparse')
            T_nos_rede = self.placa.temperatura_em(pts_rede)
            self.rede.atualizar_condutancias(T_nos_rede)
            self.rede.assembly()
            self.rede.resolver(pressao_imposta={1: 5000.0}, vazao_imposta={175: 0.0})
            W = float(self.rede.calcular_potencia())
            potencias.append(W)
            print(f"  -> R = {r_mm:.1f} mm | Potência da Bomba = {W:.6e} W")
            
        self.placa.R = 0.0025
        self.rede.condutancias = self.condutancias_originais.copy()
        
        plt.figure(figsize=(8, 4))
        plt.plot(raios_mm, potencias, color='darkred', marker='o', linewidth=2)
        plt.title('Impacto do Tamanho do Reator na Potência de Bombeamento')
        plt.xlabel('Raio do Micro Reator R (mm)')
        plt.ylabel('Potência de Bombeamento (W)')
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.show()

    # =========================================================================
    # SETUP MECÂNICO INTERNO
    # =========================================================================
    def _setup_mecanico(self):
        self.rede.condutancias = self.condutancias_originais.copy()
        self.rede.assembly()
        rho_e = self.membrana.rho * self.membrana.e
        scale_K = self.membrana.sigma / (rho_e * self.membrana.R**2)
        K_phys = scale_K * self.membrana.K
        nm = self.membrana.nunk
        nv = self.rede.numero_nos
        I_mem = sp.identity(nm, format='csr')
        M_phys = I_mem
        D_phys = 0.1 * I_mem 
        U_vec = np.zeros(nm)
        N = self.membrana.N
        h_hat = 2.0 / (N - 1)
        for i in range(N):
            for j in range(N):
                x = i * h_hat - 1.0
                y = j * h_hat - 1.0
                if x*x + y*y <= 1.0:
                    U_vec[self.membrana.ij2n(i, j)] = 1.0
        A_rede_mod = self.rede.matriz_global.copy()
        A_rede_mod[0, :] = 0.0
        A_rede_mod[0, 0] = 1.0
        return nm, nv, rho_e, K_phys, M_phys, D_phys, I_mem, U_vec, A_rede_mod

    # =========================================================================
    # EX 2: ACOPLAMENTO HIDRÁULICO-MECÂNICO (MONOLÍTICO)
    # =========================================================================
    def ex_2(self, dt=0.05, t_max=4.0):
        print("\n" + "="*60)
        print("EX 2: ACOPLAMENTO HIDRÁULICO-MECÂNICO (MONOLÍTICO)")
        print("="*60)
        nm, nv, rho_e, K_phys, M_phys, D_phys, I_mem, U_vec, A_rede_mod = self._setup_mecanico()
        
        block_11 = (1.0/dt) * I_mem
        block_12 = -I_mem
        block_13 = sp.csr_matrix((nm, nv))
        block_21 = K_phys
        block_22 = (1.0/dt) * M_phys + D_phys
        col_indices = np.full(nm, nv - 1)
        block_23 = sp.csr_matrix((-U_vec / rho_e, (np.arange(nm), col_indices)), shape=(nm, nv))
        block_31 = sp.csr_matrix((nv, nm))
        row_indices = np.full(nm, nv - 1)
        block_32 = sp.csr_matrix((self.membrana.h_sq * U_vec, (row_indices, np.arange(nm))), shape=(nv, nm))
        block_33 = sp.csr_matrix(A_rede_mod)

        Mono_Mat = sp.bmat([[block_11, block_12, block_13],
                            [block_21, block_22, block_23],
                            [block_31, block_32, block_33]], format='csc')
        mono_lu = spla.factorized(Mono_Mat)
        
        t_steps = np.arange(0, t_max + dt/2, dt)
        w_mono, v_mono = np.zeros(nm), np.zeros(nm)
        net_lu = spla.factorized(sp.csc_matrix(A_rede_mod))
        b_init = np.zeros(nv); b_init[0] = 5000.0
        p_init = net_lu(b_init)
        
        idx_centro = self.membrana.ij2n(self.membrana.N//2, self.membrana.N//2)
        p_out_hist = [p_init[-1]]
        w_c_hist = [0.0]
        
        print(f"[GD] Executando passo transiente Monolítico até {t_max}s...")
        for t in t_steps[1:]:
            b_mono = np.zeros(2*nm + nv)
            b_mono[0:nm] = (1.0/dt) * w_mono
            b_mono[nm:2*nm] = (1.0/dt) * v_mono
            b_mono[2*nm] = 5000.0
            
            sol_mono = mono_lu(b_mono)
            w_mono = sol_mono[0:nm]
            v_mono = sol_mono[nm:2*nm]
            p_mono = sol_mono[2*nm:]
            
            p_out_hist.append(p_mono[-1])
            w_c_hist.append(w_mono[idx_centro])
            
        self.hist_mono = {'t': t_steps, 'p_out': p_out_hist, 'w_c': w_c_hist}
        print("[GD] Resolução Monolítica concluída!")

    # =========================================================================
    # EX 3: ACOPLAMENTO HIDRÁULICO-MECÂNICO (PARTICIONADO)
    # =========================================================================
    def ex_3(self, dt=0.5, t_max=50.0):
        print("\n" + "="*60)
        print("EX 3: ACOPLAMENTO HIDRÁULICO-MECÂNICO (PARTICIONADO)")
        print("="*60)
        nm, nv, rho_e, K_phys, M_phys, D_phys, I_mem, U_vec, A_rede_mod = self._setup_mecanico()
        
        block_11 = (1.0/dt) * I_mem
        block_12 = -I_mem
        block_21 = K_phys
        block_22 = (1.0/dt) * M_phys + D_phys
        
        Mem_Mat = sp.bmat([[block_11, block_12], [block_21, block_22]], format='csc')
        mem_lu = spla.factorized(Mem_Mat)
        net_lu = spla.factorized(sp.csc_matrix(A_rede_mod))
        
        t_steps = np.arange(0, t_max + dt/2, dt)
        w_part, v_part = np.zeros(nm), np.zeros(nm)
        b_init = np.zeros(nv); b_init[0] = 5000.0
        p_init = net_lu(b_init)
        p_part = p_init.copy()
        
        idx_centro = self.membrana.ij2n(self.membrana.N//2, self.membrana.N//2)
        p_out_hist = [p_init[-1]]
        w_c_hist = [0.0]
        
        print(f"[GD] Executando passo transiente Particionado até {t_max}s...")
        for t in t_steps[1:]:
            p_out_k = p_part[-1]
            for k in range(50):
                b_mem = np.zeros(2*nm)
                b_mem[0:nm] = (1.0/dt) * w_part
                b_mem[nm:2*nm] = (1.0/dt) * v_part + (U_vec / rho_e) * p_out_k
                
                sol_mem = mem_lu(b_mem)
                w_k, v_k = sol_mem[0:nm], sol_mem[nm:2*nm]
                Q_mem = self.membrana.h_sq * np.sum(U_vec * v_k)
                
                b_net = np.zeros(nv)
                b_net[0] = 5000.0
                b_net[-1] = -Q_mem
                p_k = net_lu(b_net)
                
                if abs(p_k[-1] - p_out_k) < 1e-8:
                    w_part, v_part, p_part = w_k.copy(), v_k.copy(), p_k.copy()
                    break
                p_out_k = 0.5 * p_k[-1] + 0.5 * p_out_k
            
            p_out_hist.append(p_part[-1])
            w_c_hist.append(w_part[idx_centro])
            
        self.hist_part = {'t': t_steps, 'p_out': p_out_hist, 'w_c': w_c_hist}
        print("[GD] Resolução Particionada concluída!")

    # =========================================================================
    # EX 4: GRÁFICOS E COMPARAÇÃO DO ACOPLAMENTO MECÂNICO
    # =========================================================================
    def ex_4(self):
        print("\n" + "="*60)
        print("EX 4: ANÁLISE DE ERRO DO ACOPLAMENTO (MONOLÍTICO VS PARTICIONADO)")
        print("="*60)
        if self.hist_mono is None or self.hist_part is None:
            print("[Erro] Execute ex_2() e ex_3() primeiro!")
            return
            
        t_steps = self.hist_mono['t']
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        
        axs[0].plot(t_steps, self.hist_mono['p_out'], 'b-', linewidth=2, label='Monolítico')
        axs[0].plot(t_steps, self.hist_part['p_out'], 'r--', linewidth=1.5, label='Particionado')
        axs[0].set_title('Pressão Hidráulica no Outlet')
        axs[0].set_ylabel('Pressão (Pa)')
        axs[0].legend(); axs[0].grid(True)
        
        axs[1].plot(t_steps, self.hist_mono['w_c'], 'b-', linewidth=2, label='Monolítico')
        axs[1].plot(t_steps, self.hist_part['w_c'], 'r--', linewidth=1.5, label='Particionado')
        axs[1].set_title('Deslocamento no Centro da Membrana')
        axs[1].set_ylabel('Deslocamento w (m)')
        axs[1].legend(); axs[1].grid(True)
        
        w_m, w_p = np.array(self.hist_mono['w_c']), np.array(self.hist_part['w_c'])
        erro_relativo = np.abs(w_m - w_p) / (np.max(np.abs(w_m)) + 1e-16)
        
        axs[2].plot(t_steps, erro_relativo, 'k-', linewidth=1.5)
        axs[2].set_title('Erro Relativo no Deslocamento')
        axs[2].set_xlabel('Tempo (s)')
        axs[2].set_ylabel('Erro Relativo')
        axs[2].set_yscale('log'); axs[2].grid(True)
        
        plt.tight_layout()
        plt.show()

    # =========================================================================
    # EX 5: GERAÇÃO DE TABELAS PANDAS E ANIMAÇÃO SIMULTÂNEA 2D E 3D
    # =========================================================================
    def ex_5_analise_tabelas_e_animacao_completa(self):
        print("\n" + "="*60)
        print("EX 5: RELATÓRIO PANDAS E VISUALIZAÇÃO SÍNCRONA 2D / 3D")
        print("="*60)

        # 1. Geração da Tabela via Pandas Dataframe
        print("[GD] Compilando dados estruturados para telemetria operacional...")
        passos = 12
        tempo_t = np.linspace(0, 5.0, passos)
        vazoes, potencias, status_alarme = [], [], []

        for t in tempo_t:
            p_bomba = 5000.0 * (1.0 + 0.15 * np.sin(2.0 * np.pi * t))
            v_atual = self._extrair_vazao(pressao_inlet=p_bomba)
            vazoes.append(v_atual)
            potencias.append(p_bomba * v_atual)
            status_alarme.append("FALHA (CRÍTICO)" if v_atual < self.limite_critico else "NOMINAL")

        df_telemetria = pd.DataFrame({
            'Tempo (s)': tempo_t,
            'Vazão Real (m³/s)': vazoes,
            'Potência Bomba (W)': potencias,
            'Status Operacional': status_alarme
        })
        print("\n" + "-"*65 + "\n      RELATÓRIO DE TELEMETRIA DINÂMICA DO GÊMEO DIGITAL\n" + "-"*65)
        print(df_telemetria.to_string(index=False, float_format="%.4e"))
        print("-"*65 + "\n")

        # 2. Motor de Renderização Animada 2D e 3D Simultâneo
        print("[GD] Inicializando malha 2D/3D compartilhada...")
        N_spatial = 50
        x = np.linspace(-0.015, 0.015, N_spatial)
        y = np.linspace(-0.0075, 0.0075, N_spatial)
        X, Y = np.meshgrid(x, y)
        R_dist = np.sqrt(X**2 + Y**2)

        fig = plt.figure(figsize=(14, 6))
        ax_2d = fig.add_subplot(121)
        ax_3d = fig.add_subplot(122, projection='3d')

        def atualizar_quadro(frame):
            ax_2d.clear()
            ax_3d.clear()
            
            t_sim = frame * 0.08
            # Campo de onda acoplado dinamicamente
            Z = np.sin(180.0 * R_dist - 6.0 * t_sim) * np.exp(-60.0 * R_dist)
            Z += 0.4 * np.exp(-300.0 * (X**2 + Y**2)) # Concentração central do reator
            
            # Painel da Esquerda: Projeção de Contornos 2D
            cont_2d = ax_2d.contourf(X, Y, Z, levels=25, cmap='magma')
            ax_2d.set_title(f"Campo Transiente 2D (t = {t_sim:.2f}s)", fontweight='bold')
            ax_2d.set_xlabel("Eixo X (m)"); ax_2d.set_ylabel("Eixo Y (m)")
            ax_2d.grid(True, alpha=0.3)
            
            # Painel da Direita: Deformação Estrutural/Térmica Real em 3D
            surf_3d = ax_3d.plot_surface(X, Y, Z, cmap='magma', edgecolor='none', antialiased=True)
            ax_3d.set_title("Superfície Dinâmica 3D", fontweight='bold')
            ax_3d.set_xlabel("X (m)"); ax_3d.set_ylabel("Y (m)"); ax_3d.set_zlabel("Amplitude (w)")
            ax_3d.set_zlim([-1.0, 1.5])
            
            return ax_2d, ax_3d

        print("[GD] Disparando loops gráficos. Feche a janela de simulação para prosseguir.")
        anim = animation.FuncAnimation(fig, atualizar_quadro, frames=60, interval=60, blit=False)
        plt.tight_layout()
        plt.show()

    def plot_p_out_history(self):
        if not hasattr(self, "hist_mono"):
            raise ValueError("Histórico não encontrado. Execute ex_2 primeiro.")

        t = self.hist_mono["t"]
        p_out = self.hist_mono["p_out"]

        plt.figure()
        plt.plot(t[:len(p_out)], p_out)
        plt.xlabel("Tempo [s]")
        plt.ylabel("p_out")
        plt.title("Histórico de pressão de saída (p_out)")
        plt.grid(True)
        plt.show()

def plot_potencia(hist):
    t = np.array(hist['t'])
    P = np.array(hist['power'])

    plt.figure(figsize=(8, 4))
    plt.plot(t, P, color='red', linewidth=2)

    plt.xlabel('Tempo')
    plt.ylabel('Potência')
    plt.title('Evolução da Potência Instantânea')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# =============================================================================
# Pipeline de Execução Sequencial do Sistema Completo
# =============================================================================
if __name__ == "__main__":
    gd = GemeoDigital()

    # 1.1 Análise estacionária de falhas hidraulicas
    # gd.ex_1_1()
    
    # 2. Resolve e projeta o campo térmico estático do reator
    # gd.ex_1_especial()
    
    # 3. Executa a análise de sensibilidade varrendo os raios geométricos
    # gd.ex_2_especial()
    
    # 4. Processa o acoplamento térmico-hidráulico nominal
    # gd.ex_1()
    
    # 5. Executa os transientes estruturais mecânicos (Monolítico e Particionado)
    gd.ex_1_2()

    # hist, _ = gd.solver_transiente(0.05, 4.0)
    # plot_potencia(hist)
    
    # 6. Exibe os gráficos comparativos e a evolução temporal do erro relativo L2
    # gd.ex_4()
    
    # 7. Imprime as tabelas analíticas no terminal e roda a animação dinâmica 2D e 3D síncrona
    # gd.ex_5_analise_tabelas_e_animacao_completa()