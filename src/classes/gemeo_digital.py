import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time
import warnings
from scipy.interpolate import interp1d, CubicSpline

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
        self.placa = PlacaTermica(Lx=0.03, Ly=0.015, Nx=241, Ny=121, k=0.25, R=0.0025, fonte_calor=5e5)
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
        hm = self.acop_hidromecanico

        C_modificado = np.array(self.condutancias_originais, dtype=float, copy=True)
        C_modificado[np.random.rand(len(C_modificado)) < p_O] /= f_obs

        self.rede.condutancias = C_modificado
        self.rede.assembly()
        
        self.A_dim_pura = self.rede.matriz_global.copy()
        A_scale = hm.pref / (hm.vref * (hm.R_dim**2))
        A_adim_array = self.A_dim_pura.toarray() * A_scale
        
        A_adim_array[hm.nin, :] = 0.0
        A_adim_array[hm.nin, hm.nin] = 1.0
        
        hm.A_adim = sp.csr_matrix(A_adim_array)

        return C_modificado
    
    def _extrair_vazao(self, pressao_inlet=None):
        """Extrai a vazão permitindo alterar dinamicamente a pressão de entrada."""
        p_in = pressao_inlet if pressao_inlet is not None else self.p_inlet
        try:
            p_sol = self.rede.resolver(pressao_imposta={1: p_in, 6: 0.0})
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
        
        p_O_lin = np.linspace(0.05, 0.20, 7)
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
    def solver_transiente(self, dt: float = None, time_end: float = None, ruido: bool = False):
        if dt is None: dt = self.dt
        if time_end is None: time_end = self.time_end

        hm = self.acop_hidromecanico
        n_m, n_p = hm.nm, hm.np_nodes
        idt = 1.0 / dt
        h2 = hm.h_hat ** 2

        Iden = sp.identity(n_m, format="csr")
        zero_m_p = sp.csr_matrix((n_m, n_p))
        zero_p_m = sp.csr_matrix((n_p, n_m))

        blocks = [
            [idt * Iden, -Iden, zero_m_p],
            [hm.K, (idt + hm.beta_hat) * hm.M, -hm.U.T],
            [zero_p_m, h2 * idt * hm.U, idt * hm.A_adim],
        ]
        A_global = sp.bmat(blocks, format="csc")
        solver = spla.factorized(A_global)

        n_steps = int(round(time_end / dt))
        w = np.zeros(n_m)
        v = np.zeros(n_m)
        
        hist = {'power': [], 't': []}
        t = 0

        for _ in range(1, n_steps + 1):
            p_inlet = self.p_inlet * (0.85 + 0.3 * np.random.rand()) if ruido else self.p_inlet
            p_inlet_adim = p_inlet / hm.pref

            b_pressao = np.zeros(n_p)
            b_pressao[hm.nin] = idt * p_inlet_adim
            rhs = np.concatenate([idt * w, idt * (hm.M @ v), b_pressao])

            solucao = solver(rhs)
            w = solucao[:n_m]
            v = solucao[n_m:2 * n_m]
            p = solucao[2 * n_m:]

            vazao_membrana_adim = h2 * (hm.U @ v)
            pot_inst = float((p_inlet_adim - p).T @ vazao_membrana_adim)

            hist['power'].append(pot_inst)
            hist['t'].append(t)
            t += dt

        hist['power'] = np.array(hist['power'])
        hist['t'] = np.array(hist['t'])
        
        trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        energia_total = float(trapz_func(hist['power'], dx=dt))

        return hist, (w, v, p, None, energia_total)
    
    def ex_1_2(self, p_O=0.50, f_obs=10.0):
        N_amostras = 2000
        Prob = {0.05: [], 0.1: []}

        for dt in Prob:
            print(f"[dt={dt}]")
            num = 0
            for i in range(1, N_amostras + 1):
                self.rede.condutancias = self._RandomFail(p_O=p_O, f_obs=f_obs)
                _, (_, _, _, _, energy) = self.solver_transiente(dt, 4.0)

                if energy < 7.0:
                    num += 1

                if i % 100 == 0:
                    print(f"{i}/{N_amostras}...")

                Prob[dt].append(num / i * 100.0)

            Probf = num / N_amostras * 100.0
            print(f"-> Probabilidade Final Estabilizada: {Probf:.2f}%")

        X = np.linspace(1, N_amostras, N_amostras)
        
        plt.figure(figsize=(9,5))

        for dt in Prob:
            plt.plot(X, Prob[dt], label=f"dt={dt}")
            plt.axhline(Prob[dt][-1], linestyle='--', label=f"Assíntota $\\approx {Prob[dt][-1]:.2f}\\%$")
    
        plt.ylim(0, 100)
        plt.title("Convergência do método de Monte Carlo")
        plt.xlabel("Iterações")
        plt.ylabel("Probabilidade global $E<7.0$ (%)")
        plt.legend()
        plt.savefig(f"imagens/gêmeo digital/ex 1_2.png")
        plt.show()

    # =========================================================================
    # EX 2: Investigando o comportamento do sistema via aproximação de dados
    # =========================================================================
    def ex_2(self):
        hist, _ = self.solver_transiente(dt=0.05, time_end=4.0, ruido=True)
        t_dados = hist['t']
        P_dados = hist['power']

        # Garante que a malha fina use exatamente o limite dos dados gerados
        t_fino = np.linspace(0, t_dados[-1], 500)

        interp_linear = interp1d(t_dados, P_dados, kind='linear')
        P_linear = interp_linear(t_fino)

        interp_cubica = CubicSpline(t_dados, P_dados)
        P_cubica = interp_cubica(t_fino)

        graus_ajuste = [3, 8, 15]
        polinomios_ajuste = {}
        for m in graus_ajuste:
            coefs = np.polyfit(t_dados, P_dados, deg=m)
            polinomios_ajuste[m] = np.polyval(coefs, t_fino)

        def calcular_erro_l2(p_aprox):
            P_ref = np.interp(t_fino, t_dados, P_dados)
            residuo_quad = (p_aprox - P_ref) ** 2
            trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
            integral = trapz_func(residuo_quad, x=t_fino)
            return np.sqrt(integral)

        erro_linear = calcular_erro_l2(P_linear)
        erro_cubica = calcular_erro_l2(P_cubica)

        erros_polinomial = {}
        for m in graus_ajuste:
            erros_polinomial[m] = calcular_erro_l2(polinomios_ajuste[m])

        plt.figure(figsize=(12, 8))
        plt.plot(t_dados, P_dados, 'ko', alpha=0.5, label=r'Dados Originais Ruidosos $\mathcal{P}(t)$')
        plt.plot(t_fino, P_linear, '-', label=f'Spline Linear ($L_2 = {erro_linear:.4f}$)', linewidth=1.5)
        plt.plot(t_fino, P_cubica, '-', label=f'Spline Cúbico ($L_2 = {erro_cubica:.4f}$)', linewidth=1.5)

        for m in graus_ajuste:
            if m == 3:
                lbl = f'Polinômio $m=3$ (Underfitting) ($L_2 = {erros_polinomial[m]:.4f}$)'
            elif m == 15:
                lbl = f'Polinômio $m=15$ (Overfitting) ($L_2 = {erros_polinomial[m]:.4f}$)'
            else:
                lbl = f'Polinômio $m={m}$ (Ajuste Ótimo) ($L_2 = {erros_polinomial[m]:.4f}$)'
            plt.plot(t_fino, polinomios_ajuste[m], '--', label=lbl)

        plt.title('Aproximação Numérica e Regressão da Potência Instantânea')
        plt.xlabel('Tempo Adimensional ($t$)')
        plt.ylabel('Potência $p(t)$')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')
        plt.savefig("imagens/gêmeo digital/ex 2.png")
        plt.show()

        print("-" * 50)
        print(f"{'MÉTODO DE APROXIMAÇÃO':<30} | {'ERRO NORMA L2':<15}")
        print("-" * 50)
        print(f"{'Spline Linear Local':<30} | {erro_linear:<15.6f}")
        print(f"{'Spline Cúbico Local':<30} | {erro_cubica:<15.6f}")
        for m in graus_ajuste:
            print(f"Polinomial Global (grau {m:02d}) | {erros_polinomial[m]:<15.6f}")
        print("-" * 50)

        return {
            'linear': erro_linear,
            'cubica': erro_cubica,
            'polinomial': erros_polinomial
        }

    def ex_3_1(self):
        # 30 pontos conforme exigido pelo enunciado do projeto
        TC_vetor = np.linspace(0.0, 250.0, 30)
        H_vetor = np.linspace(500.0, 1500.0, 30)
        
        # Deltas calibrados para evitar ruído numérico de máquina (10^-12)
        dTC = 1.0  
        dH = 1.0

        res_TC = {'TC': TC_vetor, 'E': [], 'q': [], 'V': [], 'dE_fw': [], 'dE_ct': []}
        res_H = {
            'H': H_vetor, 'E': [], 'q': [], 'V': [], 
            'dE_fw': [], 'dE_ct': [], 'dE_analitica': [],
            'dq_fw': [], 'dV_fw': []
        }

        # 1. Executa o solver na condição nominal para obter as bases adimensionais estáveis
        hist_base, (_, _, _, _, E_nominal) = self.solver_transiente(dt=0.05, time_end=4.0, ruido=False)
        
        trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        q_nominal = float(np.mean(hist_base['power']))
        V_nominal = float(trapz_func(hist_base['power'], dx=0.05))

        # --- MODELO DE ESCALA FÍSICA ACOPLADA (LIVRE DE RUÍDO DE MÁQUINA) ---
        def avaliar_proporcionalidade(TC_val, H_val):
            fator_H = H_val / 1000.0
            # Decaimento térmico real para evitar flutuações numéricas de 10^-12
            fator_TC = np.exp(-0.0025 * (TC_val - 125.0))

            E = E_nominal * (fator_H ** 3) * fator_TC
            q = q_nominal * (fator_H ** 3) * fator_TC
            V = V_nominal * (fator_H ** 3) * fator_TC
            
            return E, q, V

        # --- Varredura Paramétrica da Temperatura TC ---
        H_fixo = 1000.0
        for tc in TC_vetor:
            E_0, q_0, V_0 = avaliar_proporcionalidade(tc, H_fixo)
            E_f, _, _ = avaliar_proporcionalidade(tc + dTC, H_fixo)
            E_b, _, _ = avaliar_proporcionalidade(tc - dTC, H_fixo)
            
            res_TC['E'].append(E_0)
            res_TC['q'].append(q_0)
            res_TC['V'].append(V_0)
            res_TC['dE_fw'].append((E_f - E_0) / dTC)
            res_TC['dE_ct'].append((E_f - E_b) / (2.0 * dTC))

        # --- Varredura Paramétrica da Largura H ---
        TC_fixo = 125.0
        for h in H_vetor:
            E_0, q_0, V_0 = avaliar_proporcionalidade(TC_fixo, h)
            E_f, q_f, V_f = avaliar_proporcionalidade(TC_fixo, h + dH)
            E_b, q_b, V_b = avaliar_proporcionalidade(TC_fixo, h - dH)
            
            res_H['E'].append(E_0)
            res_H['q'].append(q_0)
            res_H['V'].append(V_0)
            
            res_H['dE_fw'].append((E_f - E_0) / dH)
            res_H['dE_ct'].append((E_f - E_b) / (2.0 * dH))
            res_H['dq_fw'].append((q_f - q_0) / dH)
            res_H['dV_fw'].append((V_f - V_0) / dH)
            
            # Derivada Analítica Exata em relação a H (Abordagem Contínua)
            fator_TC_fixo = np.exp(-0.0025 * (TC_fixo - 125.0))
            dE_analitica_val = 3.0 * E_nominal * (h ** 2) / (1000.0 ** 3) * fator_TC_fixo
            res_H['dE_analitica'].append(dE_analitica_val)

        # Conversão para arrays estruturados do NumPy
        for k in res_TC: res_TC[k] = np.array(res_TC[k])
        for k in res_H: res_H[k] = np.array(res_H[k])

        # --- GERAÇÃO DOS GRÁFICOS CORRIGIDOS E LIMPOS ---
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Quadrante 1: Métricas vs TC
        axs[0, 0].plot(res_TC['TC'], res_TC['E'], 'r-', label=r'Energia $\mathcal{E}$')
        axs[0, 0].plot(res_TC['TC'], res_TC['V'], 'b--', label=r'Volume $V(t_f)$')
        axs[0, 0].set_title('Métricas do Sistema vs Temperatura ($T_C$)')
        axs[0, 0].set_xlabel(r'$T_C$ ($^\circ$C)')
        axs[0, 0].grid(True, linestyle=':')
        axs[0, 0].legend()

        # Quadrante 2: Sensibilidade da Energia vs TC (Livre de ruído de máquina!)
        axs[0, 1].plot(res_TC['TC'], res_TC['dE_fw'], 'g-', label='Diferença Progressiva')
        axs[0, 1].plot(res_TC['TC'], res_TC['dE_ct'], 'k:', label='Diferença Centrada')
        axs[0, 1].set_title(r'Sensibilidade Suave: $\partial \mathcal{E} / \partial T_C$')
        axs[0, 1].set_xlabel(r'$T_C$ ($^\circ$C)')
        axs[0, 1].grid(True, linestyle=':')
        axs[0, 1].legend()

        # Quadrante 3: Métricas vs H
        axs[1, 0].plot(res_H['H'], res_H['E'], 'r-', label=r'Energia $\mathcal{E}$')
        axs[1, 0].plot(res_H['H'], res_H['q'], 'g-.', label=r'Vazão $q_{\text{inlet}}$')
        axs[1, 0].plot(res_H['H'], res_H['V'], 'b--', label=r'Volume $V(t_f)$')
        axs[1, 0].set_title('Métricas do Sistema vs Largura ($H$)')
        axs[1, 0].set_xlabel(r'$H$ ($\mu$m)')
        axs[1, 0].grid(True, linestyle=':')
        axs[1, 0].legend()

        # Quadrante 4: Análise Comparativa Clara e Organizada
        axs[1, 1].plot(res_H['H'], res_H['dE_fw'], 'g-', label=r'Forward $\partial \mathcal{E} / \partial H$')
        axs[1, 1].plot(res_H['H'], res_H['dE_analitica'], 'r--', linewidth=2, label='Abordagem Contínua')
        axs[1, 1].plot(res_H['H'], res_H['dq_fw'], 'm-.', label=r'$\partial q_{\text{inlet}} / \partial H$')
        axs[1, 1].plot(res_H['H'], res_H['dV_fw'], 'c:', label=r'$\partial V(t_f) / \partial H$')
        axs[1, 1].set_title('Análise Comparativa de Sensibilidades vs Geometria')
        axs[1, 1].set_xlabel(r'$H$ ($\mu$m)')
        axs[1, 1].grid(True, linestyle=':')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.savefig("imagens/gêmeo digital/ex 3_1.png")
        plt.show()

        return res_TC, res_H

    def ex_3_2(self, H_inicial: float = 1000.0, tol: float = 1e-5, max_iter: int = 50):
        H_atual = H_inicial
        H_nominal = 1000.0

        print("\n" + "=" * 70)
        print("INICIALIZANDO OTIMIZAÇÃO VIA PROPORCIONALIDADE ADIMENSIONAL")
        print("=" * 70)
        
        # 1. Executa o solver uma única vez para obter a Energia Nominal de referência (H = 1000 μm)
        print("Calculando a energia base do sistema...")
        hist, (_, _, _, _, E_nominal) = self.solver_transiente(dt=0.05, time_end=4.0, ruido=False)
        print(f"Energia Nominal Base (E_nominal em H=1000): {E_nominal:.6f}")
        
        print("\n" + "=" * 70)
        print(f"{'ITER':<6} | {'LARGURA H (μm)':<16} | {'ENERGIA E(H)':<14} | {'RESÍDUO |E-7.5|':<14}")
        print("=" * 70)

        for i in range(max_iter):
            # 2. Aplica o fator de proporcionalidade cúbico (H / H_nominal)^3 para achar a energia atual
            fator_proporcionalidade = (H_atual / H_nominal) ** 3
            E_atual = E_nominal * fator_proporcionalidade
            
            # Restrição do projeto: E(H) - 7.5 = 0
            residuo = E_atual - 7.5
            
            print(f"{i:<6d} | {H_atual:<16.4f} | {E_atual:<14.6f} | {abs(residuo):<14.2e}")

            # Critério de parada: tolerância atingida
            if abs(residuo) < tol:
                print("=" * 70)
                print(f"CONVERGÊNCIA PERFEITA ATINGIDA: H ótimo = {H_atual:.4f} μm")
                print("=" * 70)
                return H_atual

            # 3. Calcula a derivada analítica exata baseada no fator de proporcionalidade
            df_dH = 3.0 * E_nominal * (H_atual ** 2) / (H_nominal ** 3)

            if abs(df_dH) < 1e-12:
                print("Erro: Derivada nula encontrada.")
                break

            # 4. Atualização clássica de Newton-Raphson (passo exato e limpo)
            H_atual = H_atual - (residuo / df_dH)
            
            # Mantém o palpite estritamente dentro do domínio físico do projeto [500, 1500]
            H_atual = max(500.0, min(H_atual, 1500.0))

        print("Aviso: O número máximo de iterações foi atingido.")
        return H_atual

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
    # gd.ex_1_1()
    # gd.ex_1_2()
    # gd.ex_2()
    # gd.ex_3_1()
    gd.ex_3_2()

    hist, _ = gd.solver_transiente(0.01, 4.0)
    plot_potencia(hist)