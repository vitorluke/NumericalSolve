import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from src.classes.membrana_elastica import MembranaElastica
from src.classes.rede_hidraulica import RedeHidraulica

# =============================================================================
# CONSTANTES E CONFIGURAÇÕES DO SISTEMA (CONFORME ENUNCIADO)
# =============================================================================
N_membrana = 51          # Densidade de malha adotada para consistência
R_dim = 0.25e-2          # Raio: 0.25 cm
e_dim = 0.1e-3           # Espessura: 0.1 mm
sigma_dim = 200.0        # Tensão: 200 N/m
rho_dim = 900.0          # Densidade: 900 kg/m^3
mu_fluido = 5e-4         # Viscosidade do fluido
H_k = 2000e-6            # Canais largos: 2000 µm
beta_hat = 0.0           # Sem amortecimento intrínseco
nout = 5
nin = 0

# 1. Inicialização de objetos e cálculo de escalas
membrana = MembranaElastica(N=N_membrana, R=R_dim)
nm = membrana.nunk
K = membrana.K
M = membrana.M
D = beta_hat * M  # Nula neste cenário

rede = RedeHidraulica(levels=3)
np_nodes = rede.numero_nos

# Ajuste manual da condutância hidráulica para os canais de 2000 µm
kappa_k = (np.pi * (H_k**4)) / (128 * mu_fluido)
condutancias_custom = []
for (i, j) in rede.conectividade:
    no_1 = rede.posicoes_nos[i]
    no_2 = rede.posicoes_nos[j]
    L_k = np.sqrt((no_1[0] - no_2[0])**2 + (no_1[1] - no_2[1])**2)
    condutancias_custom.append(kappa_k / L_k)
rede.condutancias = np.array(condutancias_custom)
rede.assembly()
A_dim = rede.matriz_global.copy()

# Fatores de Escala
w0 = 0.01 * R_dim
pref = (sigma_dim * w0) / (R_dim**2)
vref = np.sqrt(sigma_dim / (rho_dim * e_dim)) * (w0 / R_dim)
t_ref = R_dim * np.sqrt((rho_dim * e_dim) / sigma_dim)

A_scale = pref / (vref * (R_dim**2))
A_adimensional = A_dim * A_scale
A_adimensional[nin, :] = 0.0
A_adimensional[nin, nin] = 1.0

# Matriz de acoplamento U
U = np.zeros((np_nodes, nm))
uns = np.ones(nm)
h_hat = 2.0 / (N_membrana - 1)
for i in range(N_membrana):
    for j in range(N_membrana):
        idx = i + j * N_membrana
        if (i*h_hat - 1.0)**2 + (j*h_hat - 1.0)**2 > 1.0 or i==0 or i==N_membrana-1 or j==0 or j==N_membrana-1:
            uns[idx] = 0.0
U[nout, :] = uns
U_sparse = sp.sparse.csr_matrix(U)

# =============================================================================
# EXTRAÇÃO DO TERCEIRO MODO FUNDAMENTAL DA MEMBRANA ISOLADA
# =============================================================================
freqs_dim, omegas_dim, modes = membrana.solve_modes(nmodes=10)
freqs_hat, omegas_hat, _ = membrana.solve_modes_adimensional(nmodes=10)

# O terceiro modo válido (índice 2 devido à ordenação dos autovalores estáveis)
freq_isolada_3 = freqs_dim[2]
omega_hat_3 = omegas_hat[2]
modo_3_direcional = modes[:, 2]

print(f"--- Dados do 3º Modo da Membrana Isolada ---")
print(f"Frequência Linear Analítica/Isolada: {freq_isolada_3:.2f} Hz")
print(f"Frequência Angular Adimensional (w3_hat): {omega_hat_3:.4f}")

# =============================================================================
# EXERCÍCIO 4: Oscilação Livre a partir do 3º Modo
# =============================================================================
print("\nSimulando Exercício 4 (Oscilação Livre)...")
dt_hat = 0.01
t_final_hat = 30.0  # Tempo longo o suficiente para visualizar o amortecimento hidráulico
n_steps = int(t_final_hat / dt_hat)

# Condição Inicial: Deslocamento moldado pelo 3º modo (normalizado com amplitude pequena de 0.1)
w_current = (modo_3_direcional / np.max(np.abs(modo_3_direcional))) * 0.1
v_current = np.zeros(nm)
p_current = np.zeros(np_nodes)

# Estrutura do resolvedor Euler Implícito
idt = 1.0 / dt_hat
Iden = sp.sparse.identity(nm, format='csr')
blocks = [[idt * Iden, -Iden, None], [K, idt * M + D, -U_sparse.T], [None, U_sparse * (h_hat**2), A_adimensional]]
Aglob_LU = sp.sparse.linalg.splu(sp.sparse.bmat(blocks, format='csr'))

hist_ex4 = {'t': [], 'w_center': [], 'p_outlet': []}
idx_centro = (N_membrana // 2) + (N_membrana // 2) * N_membrana

for step in range(n_steps):
    t_hat = (step + 1) * dt_hat
    RHS_1 = idt * w_current
    RHS_2 = idt * M.dot(v_current)
    RHS_3 = np.zeros(np_nodes) # p_inlet = 0
    
    sol = Aglob_LU.solve(np.concatenate([RHS_1, RHS_2, RHS_3]))
    w_current, v_current, p_current = sol[0:nm], sol[nm:2*nm], sol[2*nm:]
    
    hist_ex4['t'].append(t_hat * t_ref)  # Tempo dimensionalizado para análise de frequência
    hist_ex4['w_center'].append(w_current[idx_centro] * w0)
    hist_ex4['p_outlet'].append(p_current[nout] * pref)

# Cálculo da Frequência Acoplada observada (via cruzamentos por zero do deslocamento central)
w_central_arr = np.array(hist_ex4['w_center'])
zero_crossings = np.where(np.diff(np.sign(w_central_arr - np.mean(w_central_arr))))[0]
if len(zero_crossings) > 2:
    tempos_cruzamento = np.array(hist_ex4['t'])[zero_crossings]
    periodo_dimensional = 2 * np.mean(np.diff(tempos_cruzamento))
    freq_acoplada_obs = 1.0 / periodo_dimensional
else:
    freq_acoplada_obs = np.nan

print(f"Frequência Observada no Sistema Acoplado: {freq_acoplada_obs:.2f} Hz")

# =============================================================================
# EXERCÍCIO 5: Carregamento Harmônico / Forçamento (Ressonância)
# =============================================================================
print("\nSimulando Exercício 5 (Forçamento Harmônico)...")
w_current = np.zeros(nm)   # Inicializa em repouso
v_current = np.zeros(nm)
p_current = np.zeros(np_nodes)

hist_ex5 = {'t_hat': [], 'w_center': [], 'p_outlet': []}

for step in range(n_steps):
    t_hat = (step + 1) * dt_hat
    
    # Pressão forçante na entrada: 5000 * cos(w3 * t)
    p_inlet_dim = 5000.0 * np.cos(omega_hat_3 * t_hat)
    p_inlet_hat = p_inlet_dim / pref
    
    RHS_1 = idt * w_current
    RHS_2 = idt * M.dot(v_current)
    RHS_3 = np.zeros(np_nodes)
    RHS_3[nin] = p_inlet_hat
    
    sol = Aglob_LU.solve(np.concatenate([RHS_1, RHS_2, RHS_3]))
    w_current, v_current, p_current = sol[0:nm], sol[nm:2*nm], sol[2*nm:]
    
    hist_ex5['t_hat'].append(t_hat)
    hist_ex5['w_center'].append(w_current[idx_centro] * w0)
    hist_ex5['p_outlet'].append(p_current[nout] * pref)

# =============================================================================
# PLOTAGEM DOS GRÁFICOS RESULTANTES
# =============================================================================
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análise Dinâmica e Vibracional (Exercícios 4 e 5)', fontsize=16, fontweight='bold')

# Ex 4: Decaimento e oscilação livre do ponto central
axs[0, 0].plot(hist_ex4['t'], hist_ex4['w_center'], 'g-')
axs[0, 0].set_title('Exercício 4: Deflexão Central (Oscilação Livre)')
axs[0, 0].set(xlabel='Tempo (s)', ylabel='Deslocamento $w_{center}$ (m)')
axs[0, 0].grid(True)

# Ex 4: Resposta de Pressão no Outlet decorrente da oscilação livre
axs[0, 1].plot(hist_ex4['t'], hist_ex4['p_outlet'], 'darkgreen')
axs[0, 1].set_title('Exercício 4: Pressão de Saída $p_{outlet}$')
axs[0, 1].set(xlabel='Tempo (s)', ylabel='Pressão (Pa)')
axs[0, 1].grid(True)

# Ex 5: Resposta harmônica mostrando o batimento/crescimento de ressonância
axs[1, 0].plot(hist_ex5['t_hat'], hist_ex5['w_center'], 'b-')
axs[1, 0].set_title('Exercício 5: Resposta Harmônica (Ressonância em $\omega_3$)')
axs[1, 0].set(xlabel='Tempo Adimensional $\hat{t}$', ylabel='Deslocamento $w_{center}$ (m)')
axs[1, 0].grid(True)

# Ex 5: Evolução da pressão transiente sob forçamento senoidal
axs[1, 1].plot(hist_ex5['t_hat'], hist_ex5['p_outlet'], 'r-')
axs[1, 1].set_title('Exercício 5: Histórico de Pressão $p_{outlet}(t)$')
axs[1, 1].set(xlabel='Tempo Adimensional $\hat{t}$', ylabel='Pressão $p_{outlet}$ (Pa)')
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()