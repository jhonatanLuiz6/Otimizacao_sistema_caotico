# Autor: @Jhonatan Luiz Souza Siqueira

# ***************************************************************************************************************/
# Eng. Controle e Automação */
# Autor: Jhonatan Luiz Souza Siqueira */
# Centro Federal de Educação e Tecnologia (CEFET-MG) */
# Monografia de Graduação */
# OTIMIZAÇÃO DO MÉTODO DE PYRAGAS PARA CONTROLE DE SISTEMAS CAÓTICOS UTILIZANDO LÓGICA FUZZY E ALGORITMO GENÉTICO */
# Orientador: Marlon José do Carmo */
# **
# Classe com métodos utilizados no algoritmo genético
# /*(AG Fuzzy-Pyragas)*/
# **************************************************************************************************************/

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random as rd

# _____________________________________________________________________________________________________________________________________________
# criar pontos das funções de pertinencia das variáveis de entrada
def f_criar_funcao_pertinencia1(ud):
    ref1 = ud[1]/2
    ref2 = ud[ud.shape[-1] -1]/2

    pts_Z = sorted([np.random.uniform(ref1, ref2), np.random.uniform(ref1, ref2), np.random.uniform(ref1, ref2)])
    Z_left = pts_Z[0]
    Z_center = pts_Z[1]
    Z_right = pts_Z[2]

    MN_left = ud[0] - 1
    MN_centerL = ud[0]
    MN_right = np.random.uniform(ud[1], Z_left)
    MN_centerR = np.random.uniform(MN_centerL, MN_right)
    pts_MN = [MN_left, MN_centerL, MN_centerR, MN_right]

    MP_right = ud[ud.shape[-1] -1] + 1
    MP_centerR = ud[ud.shape[-1] -1]
    MP_left = np.random.uniform(Z_right, ud[ud.shape[-1] - 1])
    MP_centerL = np.random.uniform(MP_left, MP_centerR)
    pts_MP = [MP_left, MP_centerL, MP_centerR, MP_right]

    N_center = np.random.uniform(MN_centerR, Z_center)
    N_right = max(MN_right, max(N_center, np.random.uniform(Z_left, Z_right)))
    N_left = min(Z_left, min(N_center, np.random.uniform(MN_centerR, MN_right)))
    pts_N = [N_left, N_center, N_right]

    P_center = np.random.uniform(Z_center, MP_centerL)
    P_left = min(MP_left, min(P_center, np.random.uniform(Z_left, Z_right)))
    P_right = max(Z_right, max(P_center, np.random.uniform(MP_left, MP_centerL)))
    pts_P = [P_left, P_center, P_right]

    return pts_MN, pts_N, pts_Z, pts_P, pts_MP

# _____________________________________________________________________________________________________________________________________________
# criar pontos das funções de pertinencia das variáveis de saída
def f_criar_funcao_pertinencia2(ud):
    ref1 = ud[0]/2
    ref2 = ud[ud.shape[-1]-1]/2

    pts_M = sorted([np.random.uniform(ref1, ref2), np.random.uniform(ref1, ref2), np.random.uniform(ref1, ref2)])
    M_left = pts_M[0]
    M_center = pts_M[1]
    M_right = pts_M[2]

    MB_left = ud[0]
    MB_right = np.random.uniform(ud[0], M_left)
    MB_center = np.random.uniform(ud[1], MB_right)
    pts_MB = sorted([MB_left, MB_center, MB_right])

    MA_right = ud[ud.shape[-1]-1]
    MA_left = np.random.uniform(M_right, ud[ud.shape[-1] -1])
    MA_center = np.random.uniform(MA_left, ud[ud.shape[-1] -1])
    pts_MA = sorted([MA_left, MA_center, MA_right])

    B_center = np.random.uniform(MB_center, M_center)
    B_right = max(MB_right, max(B_center, np.random.uniform(M_left, M_right)))
    B_left = min(M_left, min(B_center, np.random.uniform(MB_center, MB_right)))
    pts_B = [B_left, B_center, B_right]

    A_center = np.random.uniform(M_center, MA_center)
    A_left = min(MA_left, min(A_center, np.random.uniform(M_left, M_right)))
    A_right = max(M_right, max(A_center, np.random.uniform(MA_left, MA_center)))
    pts_A = [A_left, A_center, A_right]

    return pts_MB, pts_B, pts_M, pts_A, pts_MA

# _____________________________________________________________________________________________________________________________________________
def f_selecao_roleta(probabilidade):
    soma_cumulativa = np.cumsum(probabilidade)    # soma cumulativa das probabilidades de cada individuo
    num_aleatorio = sum(probabilidade)*np.random.rand()
    indice = np.argwhere(num_aleatorio <= soma_cumulativa)    # encontrar os indices dos elementos não nulos do array de probabilidade
    return indice[0][0]
# _____________________________________________________________________________________________________________________________________________
def f_cruzamento(pai1, pai2, gamma=0.1):
    filho1 = pai1.deepcopy()
    filho2 = pai2.deepcopy()

    for i in range(len(pai1.posicao)):
        alpha = np.random.uniform(-gamma, 1 + gamma, len(pai1.posicao[i]))          # criar um array com o tamanho do ponto da func pertinencia usada no cruzamento

        filho1.posicao[i] = alpha * pai1.posicao[i] + (1 - alpha) * pai2.posicao[i]
        filho2.posicao[i] = alpha * pai2.posicao[i] + (1 - alpha) * pai1.posicao[i]

        filho1.posicao[i] = filho1.posicao[i].tolist()
        filho2.posicao[i] = filho2.posicao[i].tolist()

        filho1.posicao[i] = sorted(filho1.posicao[i])
        filho2.posicao[i] = sorted(filho2.posicao[i])

    return filho1, filho2
# _____________________________________________________________________________________________________________________________________________
def f_atualizar_var_fuzzy(nome_variavel, universo_discurso, array_pontos):

    if nome_variavel == 'ganho':
        variavel_fuzzy = ctrl.Consequent(universo_discurso, nome_variavel)

        variavel_fuzzy['MB'] = fuzz.trimf(universo_discurso, array_pontos[0])
        variavel_fuzzy['B'] = fuzz.trimf(universo_discurso, array_pontos[1])
        variavel_fuzzy['M'] = fuzz.trimf(universo_discurso, array_pontos[2])
        variavel_fuzzy['A'] = fuzz.trimf(universo_discurso, array_pontos[3])
        variavel_fuzzy['MA'] = fuzz.trimf(universo_discurso, array_pontos[4])
    else:
        variavel_fuzzy = ctrl.Antecedent(universo_discurso, nome_variavel)

        variavel_fuzzy['MN'] = fuzz.trapmf(universo_discurso, array_pontos[0])
        variavel_fuzzy['N']  = fuzz.trimf(universo_discurso, array_pontos[1])
        variavel_fuzzy['Z']  = fuzz.trimf(universo_discurso, array_pontos[2])
        variavel_fuzzy['P']  = fuzz.trimf(universo_discurso, array_pontos[3])
        variavel_fuzzy['MP'] = fuzz.trapmf(universo_discurso, array_pontos[4])

    return variavel_fuzzy
# _____________________________________________________________________________________________________________________________________________
def f_mutacao(filho, taxa_mutacao, sigma):
    y = filho.deepcopy()

    for i in range(len(y.posicao)):
        flag = np.random.rand(len(y.posicao[i])) <= taxa_mutacao       # flag que indica qual gene irá sobre mutação (True ou False). True = sofre mutação
        for k in range(len(y.posicao[i])):
            if flag[k]:
                y.posicao[i][k] += sigma * np.random.uniform(-1, 1)

        y.posicao[i] = sorted(y.posicao[i])
    return y.posicao
