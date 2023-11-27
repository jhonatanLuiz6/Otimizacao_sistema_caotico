# Autor: @Jhonatan Luiz Souza Siqueira

# ***************************************************************************************************************/
# Eng. Controle e Automação */
# Autor: Jhonatan Luiz Souza Siqueira */
# Centro Federal de Educação e Tecnologia (CEFET-MG) */
# Monografia de Graduação */
# OTIMIZAÇÃO DO MÉTODO DE PYRAGAS PARA CONTROLE DE SISTEMAS CAÓTICOS UTILIZANDO LÓGICA FUZZY E ALGORITMO GENÉTICO */
# Orientador: Marlon José do Carmo */
# **
# Algoritmo de otimização do conjunto fuzzy utilziado no controle de sistema caótico para o sistema de Roosler
# Simulação para o Sistema de Roosler
# /*(AG Fuzzy-Pyragas)*/
# **************************************************************************************************************/

import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import cls_algoritmo_genetico as GA
import random as rd

# _____________________________________________________________________________________________________________________________________________
# Simulação em malha aberta

def f_malha_aberta(Parametros_Rossler):
    n = Parametros_Rossler.num_iteracoes  # número de iterações
    T = Parametros_Rossler.T   # tempo de amostragem
    tempo = Parametros_Rossler.tempo

    # variáveis de estado
    xma = np.zeros(n)
    yma = np.zeros(n)
    zma = np.zeros(n)

    # Condições Iniciais
    xma[0] = Parametros_Rossler.x[0]
    yma[0] = Parametros_Rossler.y[0]
    zma[0] = Parametros_Rossler.z[0]

    # constantes da equação de roosler
    a = Parametros_Rossler.a
    b = Parametros_Rossler.b
    c = Parametros_Rossler.c

    # Simulação
    for k in range(1, n, 1):
        xma[k] = xma[k - 1] - T * (yma[k - 1] + zma[k - 1])
        yma[k] = yma[k - 1] + T * (xma[k - 1] + a * yma[k - 1])
        zma[k] = zma[k - 1] + T * (b + xma[k - 1] * zma[k - 1] - c * zma[k - 1])
        tempo[k] = tempo[k - 1] + T

    return [xma, yma, zma]
# _____________________________________________________________________________________________________________________________________________
# Controle por Pyragas
def f_controle_pyragas(Parametros_Rossler):
    n = Parametros_Rossler.num_iteracoes  # número de iterações
    T = Parametros_Rossler.T  # tempo de amostragem
    tempo = Parametros_Rossler.tempo

    # variáveis de estado
    x_py = np.zeros(n)
    y_py = np.zeros(n)
    z_py = np.zeros(n)

    u_py = np.zeros(n)

    # Condições Iniciais
    x_py[0] = Parametros_Rossler.x[0]
    y_py[0] = Parametros_Rossler.y[0]
    z_py[0] = Parametros_Rossler.z[0]

    # constantes da equação de roosler
    a = Parametros_Rossler.a
    b = Parametros_Rossler.b
    c = Parametros_Rossler.c

    # Parâmetros do controlador
    atraso = 400
    K = 4
    Tau = 10

    for k in range(1, n):
        x_py[k] = x_py[k - 1] - T * (y_py[k - 1] + z_py[k - 1]) + T * u_py[k - 1]
        y_py[k] = y_py[k - 1] + T * (x_py[k - 1] + a * y_py[k - 1])
        z_py[k] = z_py[k - 1] + T * (b + x_py[k - 1] * z_py[k - 1] - c * z_py[k - 1])

        # control action
        if k > atraso:
            u_py[k] = K * (y_py[k - Tau] - y_py[k])  # +1 para SETPOINT

        tempo[k] = tempo[k - 1] + T

    return [x_py, y_py, z_py]

# _____________________________________________________________________________________________________________________________________________
# Controle Fuzzy-Pyragas
def f_controle_fuzzy_pyragas(Parametros_Rossler, p_erro, p_erro_dot, p_ganho, p_tau):
    # Código que pode gerar os warnings como overflow devido o universo de discurso utilizado nas entradas ou saídas do controlador fuzzy
    try:
        # _________________________Informações do sistema___________________________________________________________________

        n = Parametros_Rossler.num_iteracoes  # número de iterações
        T = Parametros_Rossler.T  # tempo de amostragem
        tempo = Parametros_Rossler.tempo  # array tempo

        # variáveis de estado
        xmf = np.zeros(n)
        ymf = np.zeros(n)
        zmf = np.zeros(n)

        # Condições Iniciais
        xmf[0] = Parametros_Rossler.x[0]
        ymf[0] = Parametros_Rossler.y[0]
        zmf[0] = Parametros_Rossler.z[0]

        # Ação de controle
        u = Parametros_Rossler.u

        # constantes da equação de lorenz
        a = Parametros_Rossler.a
        b = Parametros_Rossler.b
        c = Parametros_Rossler.c

        # Entradas do controlador Fuzzy
        erro_crisp = np.zeros(n)
        erro_dot_crisp = np.zeros(n)

        # control parameters
        atraso = 400
        Tau = 1

        # regras da lógica fuzzy
        regra1 = ctrl.Rule(p_erro['MN'] & p_erro_dot['MN'], p_ganho['MB'])
        regra2 = ctrl.Rule(p_erro['MN'] & p_erro_dot['N'], p_ganho['MB'])
        regra3 = ctrl.Rule(p_erro['MN'] & p_erro_dot['Z'], p_ganho['B'])
        regra4 = ctrl.Rule(p_erro['MN'] & p_erro_dot['P'], p_ganho['B'])
        regra5 = ctrl.Rule(p_erro['MN'] & p_erro_dot['MP'], p_ganho['M'])
        regra6 = ctrl.Rule(p_erro['N'] & p_erro_dot['MN'], p_ganho['MB'])
        regra7 = ctrl.Rule(p_erro['N'] & p_erro_dot['N'], p_ganho['B'])
        regra8 = ctrl.Rule(p_erro['N'] & p_erro_dot['Z'], p_ganho['B'])
        regra9 = ctrl.Rule(p_erro['N'] & p_erro_dot['P'], p_ganho['M'])
        regra10 = ctrl.Rule(p_erro['N'] & p_erro_dot['MP'], p_ganho['A'])
        regra11 = ctrl.Rule(p_erro['Z'] & p_erro_dot['MN'], p_ganho['B'])
        regra12 = ctrl.Rule(p_erro['Z'] & p_erro_dot['N'], p_ganho['B'])
        regra13 = ctrl.Rule(p_erro['Z'] & p_erro_dot['Z'], p_ganho['M'])
        regra14 = ctrl.Rule(p_erro['Z'] & p_erro_dot['P'], p_ganho['A'])
        regra15 = ctrl.Rule(p_erro['Z'] & p_erro_dot['MP'], p_ganho['A'])
        regra16 = ctrl.Rule(p_erro['P'] & p_erro_dot['MN'], p_ganho['B'])
        regra17 = ctrl.Rule(p_erro['P'] & p_erro_dot['N'], p_ganho['M'])
        regra18 = ctrl.Rule(p_erro['P'] & p_erro_dot['Z'], p_ganho['A'])
        regra19 = ctrl.Rule(p_erro['P'] & p_erro_dot['P'], p_ganho['A'])
        regra20 = ctrl.Rule(p_erro['P'] & p_erro_dot['MP'], p_ganho['MA'])
        regra21 = ctrl.Rule(p_erro['MP'] & p_erro_dot['MN'], p_ganho['M'])
        regra22 = ctrl.Rule(p_erro['MP'] & p_erro_dot['N'], p_ganho['A'])
        regra23 = ctrl.Rule(p_erro['MP'] & p_erro_dot['Z'], p_ganho['A'])
        regra24 = ctrl.Rule(p_erro['MP'] & p_erro_dot['P'], p_ganho['MA'])
        regra25 = ctrl.Rule(p_erro['MP'] & p_erro_dot['MP'], p_ganho['MA'])

        # regras da saida tau
        regra26 = ctrl.Rule(p_erro['MN'], p_tau['MB'])
        regra27 = ctrl.Rule(p_erro['N'], p_tau['B'])
        regra28 = ctrl.Rule(p_erro['Z'] & p_erro_dot['MN'], p_tau['B'])
        regra29 = ctrl.Rule(p_erro['Z'] & p_erro_dot['N'], p_tau['B'])
        regra30 = ctrl.Rule(p_erro['Z'] & p_erro_dot['Z'], p_tau['M'])
        regra31 = ctrl.Rule(p_erro['Z'] & p_erro_dot['P'], p_tau['A'])
        regra32 = ctrl.Rule(p_erro['Z'] & p_erro_dot['MP'], p_tau['A'])
        regra33 = ctrl.Rule(p_erro['P'], p_tau['A'])
        regra34 = ctrl.Rule(p_erro['MP'], p_tau['MA'])

        inferencia_fuzzy = ctrl.ControlSystem(
            [regra1, regra2, regra3, regra4, regra5, regra6, regra7, regra8, regra9, regra10,
             regra11, regra12, regra13, regra14, regra15, regra16, regra17, regra18, regra19, regra20,
             regra21, regra22, regra23, regra24, regra25, regra26, regra27, regra28, regra29, regra30,
             regra31, regra32, regra33, regra34
             ])
        controlador_fuzzy = ctrl.ControlSystemSimulation(inferencia_fuzzy)

        # Laço de simulação do Controlador Fuzzy-Pyragas
        for k in range(1, n, 1):
            xmf[k] = xmf[k - 1] - T * (ymf[k - 1] + zmf[k - 1])  + T * u[k - 1]
            ymf[k] = ymf[k - 1] + T * (xmf[k - 1] + a * ymf[k - 1])
            zmf[k] = zmf[k - 1] + T * (b + xmf[k - 1] * zmf[k - 1] - c * zmf[k - 1])

            # control action
            if k > atraso:
                erro_crisp[k] = (ymf[k - Tau] - ymf[k])
                erro_dot_crisp[k] = (erro_crisp[k] - erro_crisp[k - 1])

                controlador_fuzzy.input['erro'] = erro_crisp[k]
                controlador_fuzzy.input['erro_dot'] = erro_dot_crisp[k]
                controlador_fuzzy.compute()

                K = controlador_fuzzy.output['ganho']
                Tau = controlador_fuzzy.output['tau']
                Tau = round(Tau)
                if Tau <= 0:
                    Tau = 1

                u[k] = K*(ymf[k - Tau] - ymf[k])

            tempo[k] = tempo[k - 1] + T

        f_custo = np.trapz(np.abs(erro_crisp), tempo, T)
        return f_custo, xmf, ymf, zmf, tempo
    except RuntimeWarning as warning:
        # Bloco executado se um RuntimeWarning for gerado
        print('[WARNING] Aviso de tempo de execução: ',  str(warning))
        return -1, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    except:
        # Bloco executado se ocorrer qualquer outra exceção não tratada
        print("[WARNING] Ocorreu um erro inesperado na simulação da população.")
        return -1, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

# _____________________________________________________________________________________________________________________________________________
def run(Parametros_Rossler, Parametros_Fuzzy, Parametros_GA):
    # _________________________Informações do sistema___________________________________________________________________
    n           = Parametros_Rossler.num_iteracoes     # número de iterações
    T           = Parametros_Rossler.T                 # tempo de amostragem
    tempo       = Parametros_Rossler.tempo           # array tempo

    # variáveis de estado
    x = Parametros_Rossler.x
    y = Parametros_Rossler.y
    z = Parametros_Rossler.z

    # Ação de controle
    u = np.zeros(n)

    # constantes da equação de lorenz
    a = Parametros_Rossler.a
    b = Parametros_Rossler.b
    c = Parametros_Rossler.c

    # Informações do controlador Fuzzy
    # universo de discurso
    UD_erro     = Parametros_Fuzzy.UD_erro
    UD_erro_dot = Parametros_Fuzzy.UD_erro_dot
    UD_ganho    = Parametros_Fuzzy.UD_saida_ganho
    UD_tau      = Parametros_Fuzzy.UD_saida_tau

    # Parâmetros do Algoritmo Genético
    tamanho_populacao   = Parametros_GA.tamanho_populacao           # Tamanho da população
    num_geracoes        = Parametros_GA.num_geracoes                # Número de gerações
    taxa_crossover      = Parametros_GA.taxa_crossover              # probabilidade de dois individuos realizarem crossover
    num_filhos          = int(np.round(taxa_crossover*tamanho_populacao/2)*2)   # número de filhos
    taxa_mutacao        = Parametros_GA.taxa_mutacao                # Taxa de mutação
    sigma               = Parametros_GA.sigma_mutacao               # parametro para operação de mutação
    gamma               = Parametros_GA.gamma_cruzamento            # parametro para operação de cruzamento


    # Template de um indiviuo vazio
    str_individuo_vazio             = structure()
    str_individuo_vazio.posicao     = None
    str_individuo_vazio.var_fuzzy   = None
    str_individuo_vazio.custo       = None
    str_individuo_vazio.defuzz      = None

    # Melhor solução encontrada
    str_Melhor_Solucao_erro             = str_individuo_vazio.deepcopy()
    str_Melhor_Solucao_erro.custo       = np.inf
    str_Melhor_Solucao_erro_dot         = str_individuo_vazio.deepcopy()
    str_Melhor_Solucao_erro_dot.custo   = np.inf
    str_Melhor_Solucao_ganho            = str_individuo_vazio.deepcopy()
    str_Melhor_Solucao_ganho.custo      = np.inf
    str_Melhor_Solucao_tau              = str_individuo_vazio.deepcopy()
    str_Melhor_Solucao_tau.custo        = np.inf

    # Soluções temporárias
    x_temp = np.zeros(n)
    y_temp = np.zeros(n)
    z_temp = np.zeros(n)
    tempo_temp = np.zeros(n)

    print('\n[Parâmetros do sistema]\n')
    print('a = {}; b = {}; c = {}'.format(a, b, c))
    print('Condições iniciais: (x0, y0, z0) = ({}, {}, {}) '.format(x[0], y[0], z[0]))
    print('\nTempo de amostragem = ', T)
    print('Número de amostras = ', n)

    print('\n[Parâmetros do controlador Fuzzy]\n')
    print('Universo de discurso das variáveis de entrada:')
    print('erro: [{} : {}]'.format(UD_erro[0],UD_erro[UD_erro.shape[-1] - 1]))
    print('erro_dot: [{} : {}]'.format(UD_erro_dot[0], UD_erro_dot[UD_erro_dot.shape[-1] - 1]))
    print('Universo de discurso das variáveis de saída:')
    print('ganho: [{} : {}]'.format(UD_ganho[0],UD_ganho[UD_ganho.shape[-1] - 1]))
    print('tau: [{} : {}]'.format(UD_tau[0], UD_tau[UD_tau.shape[-1] - 1]))

    print('\n[Parâmetros do Algoritmo Genético]\n')
    print('Tamanho da população = ', tamanho_populacao)
    print('Número de gerações = ', num_geracoes)
    print('Taxa de crossover = ', taxa_crossover)
    print('Número de filhos = ', num_filhos)

    print('\nEXECUÇÃO DO ALGORITMO GENÉTICO\n')

    # _________________________Criar a população inicial________________________________________________________________
    print('\n___ Criação das populações iniciais ___\n\n')
    # população ds variáveis Fuzzy de entrada
    pop_erro    = str_individuo_vazio.repeat(tamanho_populacao)
    pop_ganho   = str_individuo_vazio.repeat(tamanho_populacao)

    # criação da população da variável Fuzzy erro
    print('População inicial do erro:')
    for i in range(0, tamanho_populacao):
        pts_MN, pts_N, pts_Z, pts_P, pts_MP = GA.f_criar_funcao_pertinencia1(UD_erro)

        erro = ctrl.Antecedent(UD_erro, 'erro')

        erro['MN'] = fuzz.trapmf(UD_erro, pts_MN)
        erro['N'] = fuzz.trimf(UD_erro, pts_N)
        erro['Z'] = fuzz.trimf(UD_erro, pts_Z)
        erro['P'] = fuzz.trimf(UD_erro, pts_P)
        erro['MP'] = fuzz.trapmf(UD_erro, pts_MP)

        pop_erro[i].posicao = [pts_MN, pts_N, pts_Z, pts_P, pts_MP]
        pop_erro[i].var_fuzzy = erro

        print('ind{}: {}'.format(i, pop_erro[i].posicao))
        #erro.view()

    # criação da população da variável Fuzzy erro_dot
    print('\nCriação da varivael Fuzzy do erro_dot:')
    erro_dot = ctrl.Antecedent(UD_erro_dot, 'erro_dot')
    erro_dot['MN'] = fuzz.trapmf(erro_dot.universe, [-0.6, -0.5, -0.4, -0.3])
    erro_dot['N'] = fuzz.trimf(erro_dot.universe, [-0.4, -0.2, 0])
    erro_dot['Z'] = fuzz.trimf(erro_dot.universe, [-0.2, 0, 0.2])
    erro_dot['P'] = fuzz.trimf(erro_dot.universe, [0, 0.2, 0.4])
    erro_dot['MP'] = fuzz.trapmf(erro_dot.universe, [0.2, 0.4, 0.5, 0.6])

    erro_dot.view()

    # criação da população da variável Fuzzy ganho
    print('\nPopulação inicial do ganho:')
    for i in range(0, tamanho_populacao):
        pts_MB, pts_B, pts_M, pts_A, pts_MA = GA.f_criar_funcao_pertinencia2(UD_ganho)

        metodo_defuzz = "centroid"
        ganho = ctrl.Consequent(UD_ganho, 'ganho', metodo_defuzz)

        ganho['MB'] = fuzz.trimf(UD_ganho, pts_MB)
        ganho['B'] = fuzz.trimf(UD_ganho, pts_B)
        ganho['M'] = fuzz.trimf(UD_ganho, pts_M)
        ganho['A'] = fuzz.trimf(UD_ganho, pts_A)
        ganho['MA'] = fuzz.trimf(UD_ganho, pts_MA)

        pop_ganho[i].posicao = [pts_MB, pts_B, pts_M, pts_A, pts_MA]
        pop_ganho[i].var_fuzzy = ganho
        pop_ganho[i].defuzz = metodo_defuzz

        print('ind{}: {} - Defuzzificação: {}'.format(i, pop_ganho[i].posicao, pop_ganho[i].defuzz))
        #ganho.view()

    # criação da população da variável Fuzzy ganho
    print('\nPopulação inicial do tau:')
    # metodo_defuzz = rd.choice(["centroid", "mom"])      # Método de Defuzzificação
    metodo_defuzz = "centroid"
    tau = ctrl.Consequent(UD_tau, 'tau', metodo_defuzz)

    tau['MB'] = fuzz.trimf(tau.universe, [1, 5.83, 10.66])
    tau['B'] = fuzz.trimf(tau.universe, [5.83, 10.66, 15.49])
    tau['M'] = fuzz.trimf(tau.universe, [10.66, 15.49, 20.32])
    tau['A'] = fuzz.trimf(tau.universe, [15.49, 20.32, 25.15])
    tau['MA'] = fuzz.trimf(tau.universe, [20.32, 25.15, 30])

    tau.view()

    # Calcular o Custo de cada individuo da população inicial
    print('\nCalcular custo:')
    for i in range(0, tamanho_populacao):
    #for i in range(0, tamanho_populacao):
        pop_erro[i].custo, x_temp, y_temp, z_temp, tempo = f_controle_fuzzy_pyragas(Parametros_Rossler,
                                                                                    pop_erro[i].var_fuzzy,
                                                                                    erro_dot,
                                                                                    pop_ganho[i].var_fuzzy,
                                                                                    tau)

        pop_ganho[i].custo = pop_erro[i].custo

        print('ind{}: custo: {}'.format(i,pop_erro[i].custo))
        f_plotar_grafico(x_temp, y_temp, z_temp, tempo, 'Custo da Pop Inicial. Individuo :{}'.format(i))

        # selecionar melhor custo da população inicial se não houver overflow
        if pop_erro[i].custo < str_Melhor_Solucao_erro.custo and pop_erro[i].custo >= 0:
            str_Melhor_Solucao_erro = pop_erro[i].deepcopy()
            str_Melhor_Solucao_ganho = pop_ganho[i].deepcopy()

            x, y, z = x_temp, y_temp, z_temp

    # tratar erro de overflow
    print('\nTratar erro overflow\n')
    aux = tamanho_populacao
    i = 0
    while (i < aux):
        if pop_erro[i].custo < 0:
            pop_erro.pop(i)
            pop_ganho.pop(i)
            i -= 1
            aux -= 1
        i += 1
    # _________________________Melhor Custo da população incial________________________________________________________________
    print('\nMelhor custo: ', str_Melhor_Solucao_erro.custo)
    melhor_custo = np.empty(num_geracoes+1)
    melhor_custo[0] = str_Melhor_Solucao_erro.custo
    str_Melhor_Solucao_erro.var_fuzzy.view()
    str_Melhor_Solucao_ganho.var_fuzzy.view()
    f_plotar_grafico(x, y, z, tempo, 'Resultado do controle do melhor indivíduo da população inicial')
    print('\nPopulação inicial:')
    print('erro: ', pop_erro)
    print('ganho: ',pop_ganho)

    #_________________________Main Loop - Iteração das gerações ________________________________________________________________
    print('\n___ Loop de iteração para evolução das gerações ___\n')

    for it in range(1, num_geracoes+1, 1):
        print('\nGeração: ', it)

        print('\nOperação de Seleção')
        custo_total_inverso_erro = sum(1/ind.custo for ind in pop_erro)                     # Somatório do inverso do custo para resolver problema de
                                                                                            # minimização (menor custo deve ter maior probabilidade de ser selcionado)
        prob_individuo_erro = [(1/ind.custo)/custo_total_inverso_erro for ind in pop_erro]  # probabilidade de cada infividuo ser selecionado na roleta

        pop_filhos_erro = []        # população de filhos gerados para o erro
        pop_filhos_ganho = []        # população de filhos gerados para o ganho

        for _ in range(num_filhos // 2):
            # Realizar seleção dos pais para cruzamento
            indice_selecao_pai1 = GA.f_selecao_roleta(prob_individuo_erro)
            indice_selecao_pai2 = GA.f_selecao_roleta(prob_individuo_erro)

            while (indice_selecao_pai1 == indice_selecao_pai2):
                indice_selecao_pai2 = GA.f_selecao_roleta(prob_individuo_erro)

            # pais da poúlação erro
            pai1_erro = pop_erro[indice_selecao_pai1]
            pai2_erro = pop_erro[indice_selecao_pai2]

            # pais da poúlação ganho
            pai1_ganho = pop_ganho[indice_selecao_pai1]
            pai2_ganho = pop_ganho[indice_selecao_pai2]

            # Operação de Cruzamento (Crossover)
            print('Operação de Cruzamento')
            # cruzamento da filhos erro
            filho1_erro, filho2_erro = GA.f_cruzamento(pai1_erro, pai2_erro, gamma)

            # cruzamento da filhos ganho
            filho1_ganho, filho2_ganho = GA.f_cruzamento(pai1_ganho, pai2_ganho, gamma)


            # Operação de Mutação
            print('Operação de Mutação')
            # mutação da filhos erro
            filho1_erro.posicao = GA.f_mutacao(filho1_erro, taxa_mutacao, sigma)
            filho2_erro.posicao = GA.f_mutacao(filho2_erro, taxa_mutacao, sigma)

            # mutação da filhos ganho
            filho1_ganho.posicao = GA.f_mutacao(filho1_ganho, taxa_mutacao, sigma)
            filho2_ganho.posicao = GA.f_mutacao(filho2_ganho, taxa_mutacao, sigma)

            # Atualizar variavel fuzzy dos filhos
            filho1_erro.var_fuzzy = GA.f_atualizar_var_fuzzy('erro', UD_erro, filho1_erro.posicao)
            filho2_erro.var_fuzzy = GA.f_atualizar_var_fuzzy('erro', UD_erro, filho2_erro.posicao)

            filho1_ganho.var_fuzzy = GA.f_atualizar_var_fuzzy('ganho', UD_ganho, filho1_ganho.posicao)
            filho2_ganho.var_fuzzy = GA.f_atualizar_var_fuzzy('ganho', UD_ganho, filho2_ganho.posicao)

            # Avaliar Custo dos filhos
            # Filho 1
            filho1_erro.custo, x_temp, y_temp, z_temp, tempo = f_controle_fuzzy_pyragas(Parametros_Rossler,
                                                                                        filho1_erro.var_fuzzy,
                                                                                        erro_dot,
                                                                                        filho1_ganho.var_fuzzy,
                                                                                        tau)
            filho1_ganho.custo = filho1_erro.custo

            print('Custo filho1: ', filho1_erro.custo)

            # selecionar melhor custo da população inicial se não houver overflow
            if filho1_erro.custo < str_Melhor_Solucao_erro.custo and filho1_erro.custo >= 0:
                str_Melhor_Solucao_erro = filho1_erro.deepcopy()
                str_Melhor_Solucao_ganho = filho1_ganho.deepcopy()

                x, y, z = x_temp, y_temp, z_temp


            # Filho 2
            filho2_erro.custo, x_temp, y_temp, z_temp, tempo = f_controle_fuzzy_pyragas(Parametros_Rossler,
                                                                                        filho2_erro.var_fuzzy,
                                                                                        erro_dot,
                                                                                        filho2_ganho.var_fuzzy,
                                                                                        tau)
            filho2_ganho.custo = filho2_erro.custo

            print('Custo filho2: ', filho2_erro.custo)

            # selecionar melhor custo da população inicial se não houver overflow
            if filho2_erro.custo < str_Melhor_Solucao_erro.custo and filho2_erro.custo >= 0:
                str_Melhor_Solucao_erro = filho2_erro.deepcopy()
                str_Melhor_Solucao_ganho = filho2_ganho.deepcopy()

                x, y, z = x_temp, y_temp, z_temp


            # Adicionar filhos na população de filhos se não houver erro de overflow na simulação
            if filho1_erro.custo >= 0:
                pop_filhos_erro.append(filho1_erro)
                pop_filhos_ganho.append(filho1_ganho)
            if filho2_erro.custo >= 0:
                pop_filhos_erro.append(filho2_erro)
                pop_filhos_ganho.append(filho2_ganho)

        # Merge, Sort and Select
        print('\nMerge da população dos filhos\n')
        pop_erro += pop_filhos_erro
        pop_erro = sorted(pop_erro, key=lambda x: x.custo)
        pop_erro = pop_erro[0:tamanho_populacao]

        pop_ganho += pop_filhos_ganho
        pop_ganho = sorted(pop_ganho, key=lambda x: x.custo)
        pop_ganho = pop_ganho[0:tamanho_populacao]

        print('\nPopulação da geração:', it)
        print('pop_erro: ', pop_erro)
        print('pop_ganho: ', pop_ganho)

        # Armazena o melhor custo da geração
        melhor_custo[it] = str_Melhor_Solucao_erro.custo
        #f_plotar_grafico(x, y, z, tempo, 'Melhor custo da geração {}'.format(it))

        # Imprime melhor custo da geração
        print('\nGeração {}: Melhor custo = {}\n'.format(it, melhor_custo[it]))

    f_plotar_grafico(x, y, z, tempo, 'Melhor custo da geração')

    str_Melhor_Solucao_erro.var_fuzzy.view()
    str_Melhor_Solucao_ganho.var_fuzzy.view()

    out = structure()
    out.x = x
    out.melhor_solucao_erro = str_Melhor_Solucao_erro
    out.melhor_solucao_ganho = str_Melhor_Solucao_ganho
    out.desempenho_AG = melhor_custo

    return out

# _____________________________________________________________________________________________________________________________________________
def f_plotar_grafico(p_x, p_y, p_z, p_tempo, titulo):
    # Malha fechada
    plt.plot(p_tempo, p_x, color='r', lw=2, label='x');
    plt.plot(p_tempo, p_y, color='b', lw=2, label='y');
    plt.plot(p_tempo, p_z, color='g', lw=2, label='z');
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.title(titulo)
    plt.legend()
    plt.show()


# _____________________________________________________________________________________________________________________________________________
# Definição do Problema - Parametros do sistema de Rossler

str_Parametros_Rossler = structure()             # estrutura com parametros do sistema
str_Parametros_Rossler.num_iteracoes = 4000      # número de iterações
str_Parametros_Rossler.T = 0.01                  # tempo de amostragem
str_Parametros_Rossler.tempo = np.zeros(str_Parametros_Rossler.num_iteracoes)

# variáveis de estado
str_Parametros_Rossler.x = np.zeros(str_Parametros_Rossler.num_iteracoes)
str_Parametros_Rossler.y = np.zeros(str_Parametros_Rossler.num_iteracoes)
str_Parametros_Rossler.z = np.zeros(str_Parametros_Rossler.num_iteracoes)

# Ação de controle
str_Parametros_Rossler.u = np.zeros(str_Parametros_Rossler.num_iteracoes)

# constantes da equação de lorenz
str_Parametros_Rossler.a = 0.2
str_Parametros_Rossler.b = 0.2
str_Parametros_Rossler.c = 5.7

# Condições Iniciais
str_Parametros_Rossler.x[0] = 4
str_Parametros_Rossler.y[0] = 4
str_Parametros_Rossler.z[0] = 1

# _____________________________________________________________________________________________________________________________________________
# Variavies da lógica fuzzy

str_Parametros_Fuzzy = structure()

# Universo de discurso
str_Parametros_Fuzzy.UD_erro = np.arange(-10.0,10.0, str_Parametros_Rossler.T)
str_Parametros_Fuzzy.UD_erro_dot = np.arange(-0.50,0.50, str_Parametros_Rossler.T)
str_Parametros_Fuzzy.UD_saida_ganho = np.arange(0,20.0, str_Parametros_Rossler.T)
str_Parametros_Fuzzy.UD_saida_tau = np.arange(1,30.0, str_Parametros_Rossler.T)

# Entrada do Controlador Fuzzy
str_Parametros_Fuzzy.erro = ctrl.Antecedent(str_Parametros_Fuzzy.UD_erro, 'erro')
str_Parametros_Fuzzy.erro_dot = ctrl.Antecedent(str_Parametros_Fuzzy.UD_derro, 'erro_dot')

# Saida do Controlador
str_Parametros_Fuzzy.saida_ganho = ctrl.Consequent(str_Parametros_Fuzzy.UD_saida_ganho, 'saida_ganho')
str_Parametros_Fuzzy.saida_tau = ctrl.Consequent(str_Parametros_Fuzzy.UD_saida_tau, 'saida_tau')

# regras Fuzzy
#str_regras = f_regras_fuzzy(str_Parametros_Fuzzy)

# _____________________________________________________________________________________________________________________________________________
# Parâmetros do Algoritmo Genético

str_Parametros_GA = structure()
str_Parametros_GA.tamanho_populacao = 7        # Tamanho da população
str_Parametros_GA.taxa_mutacao = 0.10           # Taxa de Mutação. Geralemente utiliza-se (100 - taxa_crossover)
str_Parametros_GA.num_geracoes = 50             # numero de gerações
str_Parametros_GA.taxa_crossover = 0.90         # probabilidade de dois indivíduos realizarem a operação de crossover
str_Parametros_GA.taxa_selecao = 0.5            # taxa de seleção da população que será mantida
str_Parametros_GA.sigma_mutacao = 0.3          # parametro usado para operação de mutação
str_Parametros_GA.gamma_cruzamento = 0.5           # parametro usado para operação de cruzamento


# _____________________________________________________________________________________________________________________________________________
# Executar GA
print('___ INÍCIO DE EXECUÇÃO ___\n\n')
print('Simulação do Sistema de Rossler')
print('Autor = Jhonatan Luiz Souza Siqueira\n\n')


out = run(str_Parametros_Rossler, str_Parametros_Fuzzy, str_Parametros_GA)

# Simulação em Malha Aberta
[x_ma, y_ma, z_ma] = f_malha_aberta(str_Parametros_Rossler)

# Simulação do controle de pyragas
[xp, yp, zp] = f_controle_pyragas(str_Parametros_Rossler)
# _____________________________________________________________________________________________________________________________________________
# Resultados
# Malha aberta
plt.plot(str_Parametros_Rossler.tempo, x_ma, color='r', lw=2, label='x');
plt.plot(str_Parametros_Rossler.tempo, y_ma, color='b', lw=2, label='y');
plt.plot(str_Parametros_Rossler.tempo, z_ma, color='g', lw=2, label='z');
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Simulação em Malha Aberta')
plt.legend()
plt.show()

# Comparação com malha fechada
plt.plot(str_Parametros_Rossler.tempo, x_ma, color='r', lw=2, label='malha aberta');
plt.plot(str_Parametros_Rossler.tempo, out.x, color='b', lw=2, label='controle fuzzy-pyragas');
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Malha aberta x Malha fechada')
plt.legend()
plt.show()

# Comparação Pyragas vs Fuzzy-Pyragas
plt.plot(str_Parametros_Rossler.tempo, xp, color='r', lw=2, label='controle Pyragas');
plt.plot(str_Parametros_Rossler.tempo, out.x, color='b', lw=2, label='controle Fuzzy-Pyragas');
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Malha aberta x Malha fechada')
plt.legend()
plt.show()

# desempenho do AG
plt.plot(out.desempenho_AG)
plt.xlim(0, str_Parametros_GA.num_geracoes)
plt.xlabel('Gerações')
plt.ylabel("Melhor Custo")
plt.title("Desempenho do AG")
plt.grid(True)
plt.show()
