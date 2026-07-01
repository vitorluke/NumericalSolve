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

        # ------------------------------------------------------------------
        # Underfitting (m = 3)
        # ------------------------------------------------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(t_dados, P_dados, 'ko', alpha=0.5,
                label=r'Dados Originais $\mathcal{P}(t)$')
        plt.plot(
            t_fino,
            polinomios_ajuste[3],
            '--',
            linewidth=2,
            label=f'Polinômio grau 3 ($L_2={erros_polinomial[3]:.4f}$)'
        )
        plt.title('Underfitting — Regressão Polinomial de Grau 3')
        plt.xlabel('Tempo Adimensional ($t$)')
        plt.ylabel('Potência $p(t)$')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.savefig("imagens/gêmeo digital/ex 2_underfitting.png")
        plt.show()

        # ------------------------------------------------------------------
        # Ajuste ótimo (m = 8)
        # ------------------------------------------------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(t_dados, P_dados, 'ko', alpha=0.5,
                label=r'Dados Originais $\mathcal{P}(t)$')
        plt.plot(
            t_fino,
            polinomios_ajuste[8],
            '--',
            linewidth=2,
            label=f'Polinômio grau 8 ($L_2={erros_polinomial[8]:.4f}$)'
        )
        plt.title('Ajuste Ótimo — Regressão Polinomial de Grau 8')
        plt.xlabel('Tempo Adimensional ($t$)')
        plt.ylabel('Potência $p(t)$')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.savefig("imagens/gêmeo digital/ex 2_otimo.png")
        plt.show()

        # ------------------------------------------------------------------
        # Overfitting (m = 15)
        # ------------------------------------------------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(t_dados, P_dados, 'ko', alpha=0.5,
                label=r'Dados Originais $\mathcal{P}(t)$')
        plt.plot(
            t_fino,
            polinomios_ajuste[15],
            '--',
            linewidth=2,
            label=f'Polinômio grau 15 ($L_2={erros_polinomial[15]:.4f}$)'
        )
        plt.title('Overfitting — Regressão Polinomial de Grau 15')
        plt.xlabel('Tempo Adimensional ($t$)')
        plt.ylabel('Potência $p(t)$')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.savefig("imagens/gêmeo digital/ex 2_overfitting.png")
        plt.show()

        # ------------------------------------------------------------------
        # Comparação dos Splines
        # ------------------------------------------------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(t_dados, P_dados, 'ko', alpha=0.5,
                label=r'Dados Originais $\mathcal{P}(t)$')

        plt.plot(
            t_fino,
            P_linear,
            linewidth=2,
            label=f'Spline Linear ($L_2={erro_linear:.4f}$)'
        )

        plt.plot(
            t_fino,
            P_cubica,
            linewidth=2,
            label=f'Spline Cúbico ($L_2={erro_cubica:.4f}$)'
        )

        plt.title('Interpolação Local por Splines')
        plt.xlabel('Tempo Adimensional ($t$)')
        plt.ylabel('Potência $p(t)$')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.savefig("imagens/gêmeo digital/ex 2_splines.png")
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
        print("\n" + "="*70)
        print("EX 3.1: ANÁLISE DE SENSIBILIDADE PARAMÉTRICA")
        print("="*70)

        hm  = self.acop_hidromecanico
        n_m = hm.nm
        n_p = hm.np_nodes
        dt  = 0.05
        T_f = 4.0
        idt = 1.0 / dt
        h2  = hm.h_hat ** 2

        A_ref = hm.A_adim.copy()

        # ------------------------------------------------------------------
        # Viscosidade
        # ------------------------------------------------------------------
        def mu(T):
            return 0.001791 / (1.0 + 0.03368 * T + 0.000221 * T**2)

        # ------------------------------------------------------------------
        # Escalonamento de A
        # ------------------------------------------------------------------
        def A_scale(fator):
            """Com Dirichlet — usada no solver."""
            A = A_ref.multiply(fator).tolil()
            A[hm.nin, :] = 0.0
            A[hm.nin, hm.nin] = 1.0
            return A.tocsr()

        def A_scale_fisica(fator):
            """Sem Dirichlet — usada para extração de vazão física."""
            return A_ref.multiply(fator).tocsr()

        def A_para_TC(TC, TC_ref=125.0):
            f = mu(TC_ref) / mu(TC)
            return A_scale(f), A_scale_fisica(f)

        def A_para_H(H_um, H_ref=1000.0):
            f = (H_um / H_ref)**4
            return A_scale(f), A_scale_fisica(f)

        # ------------------------------------------------------------------
        # Motor numérico
        # ------------------------------------------------------------------
        def _solver(A_local, A_fisica, H_um=None):
            """
            Resolve o sistema acoplado com Euler implícito.
            H_um != None  →  calcula sensibilidades analíticas dY/dH.

            Definições e expressões analíticas:
            ─────────────────────────────────────────────────────────────
            Potência:   P(t)  = (p_in - p) · (h²·U·v)
            Energia:    E     = Σ P(t)·dt   [trapézio]

            dP/dH = (p_in - p)·(h²·U·sv) - sp·(h²·U·v)   [regra produto]
            dE/dH = Σ dP/dH · dt

            q_inlet = A_fisica[nin,:] · p                  [linha sem Dirichlet]
            dq/dH   = dA_fisica/dH[nin,:] · p
                    + A_fisica[nin,:] · sp                  [regra produto]

            V(tf)  = h² · (uns · w)                        [nós interiores]
            dV/dH  = h² · (uns · sw)

            Condição de Dirichlet nas sensibilidades:
            - dA/dH tem linha nin zerada (p[nin] prescrito, independe de H)
            - sp[nin] = 0 após cada passo  (dp[nin]/dH = 0 por definição)
            ─────────────────────────────────────────────────────────────
            """
            # Reconstrói A_ref do zero com H=1000 μm garantido
            H_ref = 1000e-6  # metros
            mu_ref = 5e-4
            kappa_ref = (np.pi * H_ref**4) / (128 * mu_ref)

            C_ref = []
            for (i, j) in self.rede.conectividade:
                n1 = self.rede.posicoes_nos[i]
                n2 = self.rede.posicoes_nos[j]
                L  = np.sqrt((n1[0]-n2[0])**2 + (n1[1]-n2[1])**2)
                C_ref.append(kappa_ref / L)

            C_ref = np.array(C_ref)

            calc_sens = H_um is not None

            Iden    = sp.identity(n_m, format="csr")
            zero_mp = sp.csr_matrix((n_m, n_p))
            zero_pm = sp.csr_matrix((n_p, n_m))

            blocks = [
                [idt * Iden,  -Iden,                       zero_mp      ],
                [hm.K,        (idt + hm.beta_hat) * hm.M,  -hm.U.T     ],
                [zero_pm,     h2 * idt * hm.U,              idt * A_local],
            ]
            G      = sp.bmat(blocks, format="csc")
            solveG = spla.factorized(G)

            n_steps = int(round(T_f / dt))

            w = np.zeros(n_m);  v = np.zeros(n_m)
            sw = np.zeros(n_m); sv = np.zeros(n_m); sp_vec = np.zeros(n_p)

            p_adim = self.p_inlet / hm.pref
            b_p    = np.zeros(n_p)
            b_p[hm.nin] = idt * p_adim

            # dA/dH = (4/H)·A  [Eq. 6.16]
            # A linha nin é zerada: p[nin] é prescrito, não depende de H
            if calc_sens:
                dA_dH = A_local.multiply(4.0 / H_um).tolil()
                dA_dH[hm.nin, :] = 0.0
                dA_dH = dA_dH.tocsr()

                # Versão física (sem Dirichlet) para dq/dH
                dA_dH_fisica = A_fisica.multiply(4.0 / H_um)

            E  = 0.0
            dE = 0.0

            for _ in range(n_steps):
                # --- Estado físico ---
                rhs = np.concatenate([idt * w, idt * (hm.M @ v), b_p])
                sol = solveG(rhs)
                w   = sol[:n_m]
                v   = sol[n_m:2*n_m]
                p   = sol[2*n_m:]

                qmem = h2 * (hm.U @ v)
                pot  = float((p_adim - p) @ qmem)
                E   += pot * dt

                # --- Sensibilidades em relação a H ---
                if calc_sens:
                    rhs_s = np.concatenate([
                        idt * sw,
                        idt * (hm.M @ sv),
                        -(idt * dA_dH) @ p     # <<< CORREÇÃO: G tem bloco idt*A(H) ⇒ dG/dH = idt·dA/dH
                    ])
                    sols   = solveG(rhs_s)
                    sw     = sols[:n_m]
                    sv     = sols[n_m:2*n_m]
                    sp_vec = sols[2*n_m:]

                    # dp[nin]/dH = 0: p[nin] prescrito não depende de H
                    sp_vec[hm.nin] = 0.0

                    # dP/dH = (p_in - p)·(h²·U·sv) - sp·(h²·U·v)
                    qmem_s = h2 * (hm.U @ sv)
                    dP = float((p_adim - p) @ qmem_s) - float(sp_vec @ qmem)
                    dE += dP * dt

            # --- Grandezas no instante final ---
            V_f  = float(h2 * np.dot(hm.uns, w))
            q_in = float((A_fisica[hm.nin, :] @ p).sum())

            if calc_sens:
                dV = float(h2 * np.dot(hm.uns, sw))
                dq = float((dA_dH_fisica[hm.nin, :] @ p).sum()
                        + (A_fisica[hm.nin, :]       @ sp_vec).sum())
                return E, q_in, V_f, dE, dq, dV

            return E, q_in, V_f

        # ------------------------------------------------------------------
        # VARREDURA EM TC  (H = 1000 μm fixo)
        # ------------------------------------------------------------------
        print("\nVarrendo TC ∈ [0, 250] °C ...")
        TC_vec = np.linspace(0.0, 250.0, 30)
        dTC    = 1.0

        E_TC = []; q_TC = []; V_TC = []
        dE_fw_TC = []; dE_ct_TC = []
        dq_fw_TC = []; dq_ct_TC = []
        dV_fw_TC = []; dV_ct_TC = []

        for TC in TC_vec:
            Al,  Af  = A_para_TC(TC)
            Alf, Aff = A_para_TC(TC + dTC)
            Alb, Afb = A_para_TC(TC - dTC)

            E0, q0, V0 = _solver(Al,  Af)
            Ef, qf, Vf = _solver(Alf, Aff)
            Eb, qb, Vb = _solver(Alb, Afb)

            E_TC.append(E0); q_TC.append(q0); V_TC.append(V0)
            dE_fw_TC.append((Ef - E0) / dTC)
            dE_ct_TC.append((Ef - Eb) / (2 * dTC))
            dq_fw_TC.append((qf - q0) / dTC)
            dq_ct_TC.append((qf - qb) / (2 * dTC))
            dV_fw_TC.append((Vf - V0) / dTC)
            dV_ct_TC.append((Vf - Vb) / (2 * dTC))

        TC_vec   = np.array(TC_vec)
        E_TC     = np.array(E_TC);   q_TC = np.array(q_TC);  V_TC = np.array(V_TC)
        dE_fw_TC = np.array(dE_fw_TC); dE_ct_TC = np.array(dE_ct_TC)
        dq_fw_TC = np.array(dq_fw_TC); dq_ct_TC = np.array(dq_ct_TC)
        dV_fw_TC = np.array(dV_fw_TC); dV_ct_TC = np.array(dV_ct_TC)

        # ------------------------------------------------------------------
        # VARREDURA EM H  (TC = 125 °C fixo)
        # ------------------------------------------------------------------
        print("Varrendo H ∈ [500, 1500] μm ...")
        H_vec = np.linspace(500.0, 1500.0, 30)
        dH    = 1.0

        E_H = []; q_H = []; V_H = []
        dE_fw_H = []; dE_ct_H = []
        dE_an_H = []; dq_an_H = []; dV_an_H = []
        dq_fw_H = []; dV_fw_H = []

        for H in H_vec:
            Al,  Af  = A_para_H(H)
            Alf, Aff = A_para_H(H + dH)
            Alb, Afb = A_para_H(H - dH)

            E0, q0, V0, dE_a, dq_a, dV_a = _solver(Al, Af, H_um=H)
            Ef, qf, Vf = _solver(Alf, Aff)
            Eb, _,  _  = _solver(Alb, Afb)

            E_H.append(E0);  q_H.append(q0);  V_H.append(V0)
            dE_fw_H.append((Ef - E0) / dH)
            dE_ct_H.append((Ef - Eb) / (2 * dH))
            dq_fw_H.append((qf - q0) / dH)
            dV_fw_H.append((Vf - V0) / dH)
            dE_an_H.append(dE_a)
            dq_an_H.append(dq_a)
            dV_an_H.append(dV_a)

        H_vec    = np.array(H_vec)
        E_H      = np.array(E_H);   q_H = np.array(q_H);  V_H = np.array(V_H)
        dE_fw_H  = np.array(dE_fw_H);  dE_ct_H = np.array(dE_ct_H)
        dE_an_H  = np.array(dE_an_H)
        dq_fw_H  = np.array(dq_fw_H);  dq_an_H = np.array(dq_an_H)
        dV_fw_H  = np.array(dV_fw_H);  dV_an_H = np.array(dV_an_H)

        for H in H_vec:
            Al,  Af  = A_para_H(H)
            Alf, Aff = A_para_H(H + dH)
            Alb, Afb = A_para_H(H - dH)

            E0, q0, V0, dE_a, dq_a, dV_a = _solver(Al, Af, H_um=H)
            Ef, qf, Vf = _solver(Alf, Aff)
            Eb, _,  _  = _solver(Alb, Afb)

            # ---- DIAGNÓSTICO: imprime só no primeiro ponto e para ----
            print(f"\n--- DIAGNÓSTICO H={H:.1f} μm ---")
            print(f"E0={E0:.6f}  Ef={Ef:.6f}  Eb={Eb:.6f}")
            print(f"dE FWD = {(Ef-E0)/dH:.8f}")
            print(f"dE CTD = {(Ef-Eb)/(2*dH):.8f}")
            print(f"dE AN  = {dE_a:.8f}")
            print(f"V0={V0:.8f}  Vf={Vf:.8f}  Vb={Vb:.8f}")
            print(f"dV FWD = {(Vf-V0)/dH:.10f}")
            print(f"dV CTD = {(Vf-Eb)/(2*dH):.10f}")
            print(f"dV AN  = {dV_a:.10f}")
            break  # remove após diagnóstico
            # ---------------------------------------------------------

        # ------------------------------------------------------------------
        # GRÁFICOS
        # ------------------------------------------------------------------
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle('EX 3.1 — Análise de Sensibilidade Paramétrica',
                    fontsize=14, fontweight='bold')

        # (0,0) Métricas vs TC
        axs[0,0].plot(TC_vec, E_TC,  'r-',  label=r'$\mathcal{E}$')
        axs[0,0].plot(TC_vec, q_TC,  'g-.', label=r'$q_{\rm inlet}$')
        axs[0,0].plot(TC_vec, V_TC,  'b--', label=r'$V(t_f)$')
        axs[0,0].set_title(r'Métricas vs $T_C$')
        axs[0,0].set_xlabel(r'$T_C$ (°C)')
        axs[0,0].legend(); axs[0,0].grid(True, linestyle=':')

        # (0,1) Derivadas vs TC
        axs[0,1].plot(TC_vec, dE_fw_TC, 'r-',  lw=3, alpha=0.6, label=r'FWD $\partial\mathcal{E}/\partial T_C$')
        axs[0,1].plot(TC_vec, dE_ct_TC, 'r--', lw=2,             label=r'CTD $\partial\mathcal{E}/\partial T_C$')
        axs[0,1].plot(TC_vec, dq_fw_TC, 'g-',  lw=3, alpha=0.6, label=r'FWD $\partial q/\partial T_C$')
        axs[0,1].plot(TC_vec, dq_ct_TC, 'g--', lw=2,             label=r'CTD $\partial q/\partial T_C$')
        axs[0,1].plot(TC_vec, dV_fw_TC, 'b-',  lw=3, alpha=0.6, label=r'FWD $\partial V/\partial T_C$')
        axs[0,1].plot(TC_vec, dV_ct_TC, 'b--', lw=2,             label=r'CTD $\partial V/\partial T_C$')
        axs[0,1].set_title(r'Derivadas vs $T_C$ (diferenças finitas)')
        axs[0,1].set_xlabel(r'$T_C$ (°C)')
        axs[0,1].legend(fontsize=7); axs[0,1].grid(True, linestyle=':')

        # (1,0) Métricas vs H
        axs[1,0].plot(H_vec, E_H,  'r-',  label=r'$\mathcal{E}$')
        axs[1,0].plot(H_vec, q_H,  'g-.', label=r'$q_{\rm inlet}$')
        axs[1,0].plot(H_vec, V_H,  'b--', label=r'$V(t_f)$')
        axs[1,0].set_title(r'Métricas vs $H$')
        axs[1,0].set_xlabel(r'$H$ (μm)')
        axs[1,0].legend(); axs[1,0].grid(True, linestyle=':')

        # (1,1) dE/dH: numérico vs analítico
        axs[1,1].plot(H_vec, dE_fw_H, 'g-',  lw=4, alpha=0.5, label=r'FWD $\partial\mathcal{E}/\partial H$')
        axs[1,1].plot(H_vec, dE_ct_H, 'g--', lw=2,             label=r'CTD $\partial\mathcal{E}/\partial H$')
        axs[1,1].plot(H_vec, dE_an_H, 'r-',  lw=2,             label=r'Analítico (Eq. 6.17 adaptada)')
        axs[1,1].set_title(r'$\partial\mathcal{E}/\partial H$: Numérico vs Analítico')
        axs[1,1].set_xlabel(r'$H$ (μm)')
        axs[1,1].legend(); axs[1,1].grid(True, linestyle=':')

        # (2,0) dq/dH: numérico vs analítico
        axs[2,0].plot(H_vec, dq_fw_H, 'm-',  lw=4, alpha=0.5, label=r'FWD $\partial q/\partial H$')
        axs[2,0].plot(H_vec, dq_an_H, 'k--', lw=2,             label=r'Analítico $\partial q/\partial H$')
        axs[2,0].set_title(r'$\partial q_{\rm inlet}/\partial H$: Numérico vs Analítico')
        axs[2,0].set_xlabel(r'$H$ (μm)')
        axs[2,0].legend(); axs[2,0].grid(True, linestyle=':')

        # (2,1) dV/dH: numérico vs analítico
        axs[2,1].plot(H_vec, dV_fw_H, 'c-',  lw=4, alpha=0.5, label=r'FWD $\partial V/\partial H$')
        axs[2,1].plot(H_vec, dV_an_H, 'b--', lw=2,             label=r'Analítico $\partial V/\partial H$')
        axs[2,1].set_title(r'$\partial V(t_f)/\partial H$: Numérico vs Analítico')
        axs[2,1].set_xlabel(r'$H$ (μm)')
        axs[2,1].legend(); axs[2,1].grid(True, linestyle=':')

        plt.tight_layout()
        plt.savefig("imagens/gêmeo digital/ex 3_1.png")
        plt.show()

        return {
            'TC': (TC_vec, E_TC, q_TC, V_TC,
                dE_fw_TC, dE_ct_TC,
                dq_fw_TC, dq_ct_TC,
                dV_fw_TC, dV_ct_TC),
            'H':  (H_vec, E_H, q_H, V_H,
                dE_fw_H, dE_ct_H, dE_an_H,
                dq_fw_H, dq_an_H,
                dV_fw_H, dV_an_H)
        }

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
    gd.ex_2()
    # gd.ex_3_1()
    # gd.ex_3_2()

    hist, _ = gd.solver_transiente(0.01, 4.0)
    plot_potencia(hist)