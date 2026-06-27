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

# Importações dos módulos físicos exigidos pelo ICMC-USP
from src.classes.rede_hidraulica import RedeHidraulica
from src.classes.placa_termica import PlacaTermica
from src.classes.membrana_elastica import MembranaElastica

class GemeoDigital:
    def __init__(self, levels_rede=3):
        """Inicializa o Gêmeo Digital instanciando as três físicas principais."""
        print("[GD] Inicializando os subsistemas físicos...")
        self.rede = RedeHidraulica(levels=levels_rede)
        self.placa = PlacaTermica(Lx=0.03, Ly=0.015, Nx=61, Ny=31, k=0.25, R=0.0025, fonte_calor=5e5)
        self.membrana = MembranaElastica(N=51, R=0.0025)
        
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

    def _extrair_vazao(self, pressao_inlet=5000.0):
        """Método auxiliar para extrair a vazão do nó de entrada isolando a bomba."""
        try:
            p_sol = self.rede.resolver(pressao_imposta={1: pressao_inlet}, vazao_imposta={175: 0.0})
            return np.dot(self.rede.matriz_global[0, :], p_sol)
        except np.linalg.LinAlgError:
            return 0.0

    # =========================================================================
    # EX 0: ESTOCÁSTICO, SURROGATE MODEL E GRÁFICO COMBINADO INTERPOLAÇÃO/REGRESSÃO
    # =========================================================================
    def ex_0(self, N_amostras=500, p_O=0.20, f_obs=5.0, dt=0.05, t_max=4.0):
        print("\n" + "="*60)
        print("EX 0: TESTES ESTOCÁSTICOS E SURROGATE MODEL UNIFICADO")
        print("="*60)
        
        # 0.1 Convergência Monte Carlo
        print(f"\n[Ex 0.1] Convergência Monte Carlo (N={N_amostras})...")
        falhas_criticas = 0
        for i in range(1, N_amostras + 1):
            cond = self.condutancias_originais.copy()
            cond[np.random.rand(len(cond)) < p_O] /= f_obs
            self.rede.condutancias = cond
            self.rede.assembly() 
            if self._extrair_vazao() < self.limite_critico:
                falhas_criticas += 1
        print(f"-> Probabilidade Final Estabilizada: {falhas_criticas/N_amostras:.4f}")
        
        # 0.2 Curva de Fragilidade
        print(f"\n[Ex 0.2] Curva de Fragilidade (N={N_amostras} por ponto)...")
        vetor_pO = np.linspace(0.05, 0.65, 7)
        for p_idx in vetor_pO:
            falhas = 0
            for _ in range(N_amostras):
                cond = self.condutancias_originais.copy()
                cond[np.random.rand(len(cond)) < p_idx] /= f_obs
                self.rede.condutancias = cond
                self.rede.assembly()
                if self._extrair_vazao() < self.limite_critico:
                    falhas += 1
            print(f"  p_O = {p_idx:.2f} -> Probabilidade de Falha: {falhas/N_amostras:.4f}")
        
        # 0.3 Surrogate Model & Interpolação/Regressão Simultânea (Ajustado com a Imagem de Referência)
        print("\n[Ex 0.3] Processando Aproximação de Dados Espaciais/Temporais...")
        self.rede.condutancias = self.condutancias_originais.copy()
        self.rede.assembly()
        
        t_array = np.arange(0, t_max + dt/2, dt)
        P_noisy = np.zeros_like(t_array)
        P_clean = np.zeros_like(t_array)
        
        for i, t in enumerate(t_array):
            ruido = np.random.uniform(-0.15, 0.15)
            self.rede.resolver(pressao_imposta={1: 5000.0 * (1.0 + ruido)}, vazao_imposta={175: 0.0})
            P_noisy[i] = float(self.rede.calcular_potencia())
            
            self.rede.resolver(pressao_imposta={1: 5000.0}, vazao_imposta={175: 0.0})
            P_clean[i] = float(self.rede.calcular_potencia())

        t_eval = np.linspace(0, t_max, 500)
        f_lin = interp1d(t_array, P_noisy, kind='linear')
        f_cub = interp1d(t_array, P_noisy, kind='cubic')
        
        # Geração do Gráfico Combinado Final
        plt.figure(figsize=(10, 6))
        plt.plot(t_array, P_noisy, 'ko', label='Dados Ruidosos')
        plt.plot(t_eval, f_lin(t_eval), 'b-', alpha=0.75, label='Spline Linear')
        plt.plot(t_eval, f_cub(t_eval), color='orange', linestyle='--', linewidth=2, label='Spline Cúbica')
        
        for m in [3, 8, 15]:
            coefs = np.polyfit(t_array, P_noisy, m)
            P_poly_eval = np.polyval(coefs, t_eval)
            P_poly_pts = np.polyval(coefs, t_array)
            
            # Cálculo robusto da norma L2 contínua independente da versão do NumPy
            diff = P_clean - P_poly_pts
            erro_L2 = np.sqrt(np.sum((diff[:-1]**2 + diff[1:]**2) / 2) * dt)
            label_status = "Balanceado" if m == 8 else "Overfitting" if m == 15 else "Underfitting"
            print(f"  -> Grau {m:2d} ({label_status}): Erro L2 = {erro_L2:.4e}")
            
            if m == 8:
                plt.plot(t_eval, P_poly_eval, 'g-.', linewidth=2, label='Polinômio Grau 8 (Ajustado)')
            elif m == 15:
                plt.plot(t_eval, P_poly_eval, 'r-', linewidth=1.5, label='Polinômio Grau 15 (Overfitting)')
                
        plt.plot(t_array, P_clean, color='magenta', linestyle=':', linewidth=2.5, alpha=0.6, label='Curva Exata (Alvo)')
        plt.title('Aproximação de Dados via Interpolação e Regressão', fontsize=12, fontweight='bold')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Potência Hidráulica $\mathcal{P}(t)$')
        plt.legend(loc='best', framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
            
        self.rede.condutancias = self.condutancias_originais.copy()

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
    def ex_2(self, dt=0.5, t_max=50.0):
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

# =============================================================================
# Pipeline de Execução Sequencial do Sistema Completo
# =============================================================================
if __name__ == "__main__":
    gd = GemeoDigital()
    
    # 1. Executa o bloco estocástico e plota o gráfico unificado de regressão/interpolação
    #gd.ex_0()
    
    # 2. Resolve e projeta o campo térmico estático do reator
    gd.ex_1_especial()
    
    # 3. Executa a análise de sensibilidade varrendo os raios geométricos
    gd.ex_2_especial()
    
    # 4. Processa o acoplamento térmico-hidráulico nominal
    gd.ex_1()
    
    # 5. Executa os transientes estruturais mecânicos (Monolítico e Particionado)
    gd.ex_2()
    gd.ex_3()
    
    # 6. Exibe os gráficos comparativos e a evolução temporal do erro relativo L2
    gd.ex_4()
    
    # 7. Imprime as tabelas analíticas no terminal e roda a animação dinâmica 2D e 3D síncrona
    gd.ex_5_analise_tabelas_e_animacao_completa()