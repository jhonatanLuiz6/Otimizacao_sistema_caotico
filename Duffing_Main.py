
# Autor: @Jhonatan Luiz Souza Siqueira

# ***************************************************************************************************************/
# Eng. Controle e Automação */
# Autor: Jhonatan Luiz Souza Siqueira */
# Centro Federal de Educação e Tecnologia (CEFET-MG) */
# Monografia de Graduação */
# OTIMIZAÇÃO DO MÉTODO DE PYRAGAS PARA CONTROLE DE SISTEMAS CAÓTICOS UTILIZANDO LÓGICA FUZZY E ALGORITMO GENÉTICO */
# Orientador: Marlon José do Carmo */
# **
# Algoritmo de otimização do conjunto fuzzy utilziado no controle de sistema caótico para o mapa logístico
# Simulação para o Oscilador de Duffing
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
def f_malha_aberta(Parametros_Duffing):
    n = Parametros_Duffing.num_iteracoes  # número de iterações
    T = Parametros_Duffing.T  # tempo de amostragem
    tempo_ma = np.zeros(n)

    x1ma = np.zeros(n)
    x2ma = np.zeros(n)
    yma = np.zeros(n)

    x1ma[0] = Parametros_Duffing.x1[0]
    x2ma[0] = Parametros_Duffing.x2[0]

    # Constantes do oscilador de duffing
    alpha   = Parametros_Duffing.alpha
    beta    = Parametros_Duffing.beta
    omega   = Parametros_Duffing.omega
    delta   = Parametros_Duffing.delta
    gama    = Parametros_Duffing.gama

    for k in range(1, n, 1):
        x1ma[k] = x1ma[k - 1] + T * x2ma[k - 1]

        x2ma[k] = (-alpha * T) * x1ma[k - 1] + (-delta * T + 1) * x2ma[k - 1] + gama * np.cos(omega * k * T) - (
                    beta * T) * x1ma[k - 1] * x1ma[k - 1] * x1ma[k - 1]

        yma[k] = x2ma[k]
        tempo_ma[k] = tempo_ma[k - 1] + T

    return x1ma, x2ma
# _____________________________________________________________________________________________________________________________________________
def f_controle_pyragas(Parametros_Duffing):
    n = Parametros_Duffing.num_iteracoes  # número de iterações
    T = Parametros_Duffing.T  # tempo de amostragem
    tempo_p = np.zeros(n)

    x1_p = np.zeros(n)
    x2_p = np.zeros(n)
    y_p = np.zeros(n)

    u = np.zeros(n)

    x1_p[0] = Parametros_Duffing.x1[0]
    x2_p[0] = Parametros_Duffing.x2[0]

    # Constantes do oscilador de duffing
    alpha = Parametros_Duffing.alpha
    beta = Parametros_Duffing.beta
    omega = Parametros_Duffing.omega
    delta = Parametros_Duffing.delta
    gama = Parametros_Duffing.gama

    # Parâmetros do controle
    atraso = 10
    Tau = 39
    K = 27.95

    for k in range(1, n, 1):
        x1_p[k] = x1_p[k - 1] + T * x2_p[k - 1]

        x2_p[k] = (-alpha * T) * x1_p[k - 1] + (-delta * T + 1) * x2_p[k - 1] + gama * np.cos(omega * k * T) - (
                beta * T) * x1_p[k - 1] * x1_p[k - 1] * x1_p[k - 1]  + T*u[k-1]

        y_p[k] = x2_p[k]
        # ação de controle
        if k > atraso:
            u[k] = K * (x2_p[k - Tau] - x2_p[k])

        tempo_p[k] = tempo_p[k - 1] + T

    return x1_p, x2_p
# _____________________________________________________________________________________________________________________________________________
def f_controle_fuzzy_pyragas(Parametros_Duffing, p_erro, p_erro_dot, p_ganho, p_tau):
    # Código que pode gerar os warnings como overflow devido o universo de discurso utilizado nas entradas ou saídas do controlador fuzzy
    try:
        # _________________________Informações do sistema___________________________________________________________________

        n = Parametros_Duffing.num_iteracoes  # número de iterações
        T = Parametros_Duffing.T  # tempo de amostragem
        tempo = Parametros_Duffing.tempo  # array tempo

        # variáveis de estado
        x1mf = np.zeros(n)
        x2mf = np.zeros(n)
        ymf = np.zeros(n)

        # Condições Iniciais
        x1mf[0] = Parametros_Duffing.x1[0]
        x2mf[0] = Parametros_Duffing.x2[0]

        # Ação de controle
        u = np.zeros(n)

        # constantes da equação de lorenz
        alpha = Parametros_Duffing.alpha
        beta = Parametros_Duffing.beta
        omega = Parametros_Duffing.omega
        delta = Parametros_Duffing.delta
        gama = Parametros_Duffing.gama

        # Entradas do controlador Fuzzy
        erro_crisp = np.zeros(n)
        erro_dot_crisp = np.zeros(n)

        # control parameters
        atraso = 1000
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

        # Laço de simulação do Controlador Fuzzy
        for k in range(1, n, 1):
            x1mf[k] = x1mf[k - 1] + T * x2mf[k - 1]

            x2mf[k] = (-alpha * T) * x1mf[k - 1] + (-delta * T + 1) * x2mf[k - 1] + gama * np.cos(omega * k * T) - (
                    beta * T) * x1mf[k - 1] * x1mf[k - 1] * x1mf[k - 1] + T*u[k-1]

            ymf[k] = x2mf[k]
            # ação de controle
            if k > atraso:
                erro_crisp[k] = ymf[k - Tau] - ymf[k]
                erro_dot_crisp[k] = (erro_crisp[k] - erro_crisp[k - 1])

                controlador_fuzzy.input['erro'] = erro_crisp[k]
                controlador_fuzzy.input['erro_dot'] = erro_dot_crisp[k]
                controlador_fuzzy.compute()

                K = controlador_fuzzy.output['ganho']
                Tau = controlador_fuzzy.output['tau']
                Tau = round(Tau)
                if Tau <= 0:
                    Tau = 1

                u[k] = K * (ymf[k - Tau] - ymf[k])

            tempo[k] = tempo[k - 1] + T

        f_custo = np.trapz(np.abs(erro_crisp), tempo, T)
        return f_custo, x1mf, x2mf, tempo
    except RuntimeWarning as warning:
        # Bloco executado se um RuntimeWarning for gerado
        print('[WARNING] Aviso de tempo de execução: ',  str(warning))
        return -1, np.zeros(n), np.zeros(n), np.zeros(n)
    except:
        # Bloco executado se ocorrer qualquer outra exceção não tratada
        print("[WARNING] Ocorreu um erro inesperado na simulação da população.")
        return -1, np.zeros(n), np.zeros(n), np.zeros(n)

# _____________________________________________________________________________________________________________________________________________
def run(Parametros_Duffing, Parametros_Fuzzy, Parametros_GA):
    # _________________________Informações do sistema___________________________________________________________________
    n           = Parametros_Duffing.num_iteracoes     # número de iterações
    T           = Parametros_Duffing.T                 # tempo de amostragem
    tempo       = Parametros_Duffing.tempo           # array tempo

    # variáveis de estado
    x1 = Parametros_Duffing.x1
    x2 = Parametros_Duffing.x2
    y = Parametros_Duffing.y

    # Ação de controle
    u = np.zeros(n)

    # constantes da equação de lorenz
    alpha = Parametros_Duffing.alpha
    beta = Parametros_Duffing.beta
    omega = Parametros_Duffing.omega
    delta = Parametros_Duffing.delta
    gama = Parametros_Duffing.gama

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
    simga_mutacao       = Parametros_GA.sigma_mutacao  # parametro para operação de mutação
    gamma_cruzamento    = Parametros_GA.gamma_cruzamento  # parametro para operação de cruzamento

    # Template de um indiviuo vazio
    str_individuo_vazio             = structure()
    str_individuo_vazio.posicao     = None
    str_individuo_vazio.var_fuzzy   = None
    str_individuo_vazio.custo       = None
    str_individuo_vazio.defuzz      = None

    # Melhor solução encontrada
    str_Melhor_Solucao_erro          = str_individuo_vazio.deepcopy()
    str_Melhor_Solucao_erro.custo    = np.inf
    str_Melhor_Solucao_erro_dot = str_individuo_vazio.deepcopy()
    str_Melhor_Solucao_erro_dot.custo = np.inf
    str_Melhor_Solucao_ganho = str_individuo_vazio.deepcopy()
    str_Melhor_Solucao_ganho.custo = np.inf
    str_Melhor_Solucao_tau = str_individuo_vazio.deepcopy()
    str_Melhor_Solucao_tau.custo = np.inf

    # Soluções temporárias
    x1_temp = np.zeros(n)
    x2_temp = np.zeros(n)
    tempo_temp = np.zeros(n)

    print('\n[Parâmetros do sistema]\n')
    print('alpha = {}; beta = {}; omega = {}; delta = {}; gama = {}'.format(alpha, beta, omega, delta, gama))
    print('Condições iniciais: [x1(0), x2(0)] = [{}, {}] '.format(x1[0], x2[0]))
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
    # população da variável Fuzzy
    pop_erro = str_individuo_vazio.repeat(tamanho_populacao)
    pop_ganho = str_individuo_vazio.repeat(tamanho_populacao)

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
    print('\nCriação da variável fuzzy do ganho:')
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
        # ganho.view()

    # criação da população da variável Fuzzy Tau
    print('\nCriação da população da variável Fuzzy tau:')

    #metodo_defuzz = rd.choice(["centroid", "mom"])      # Método de Defuzzificação
    metodo_defuzz = "centroid"
    tau = ctrl.Consequent(UD_tau, 'tau', metodo_defuzz)

    tau['MB'] = fuzz.trimf(tau.universe, [0, 8.0, 15])
    tau['B'] = fuzz.trimf(tau.universe, [8, 13, 20])
    tau['M'] = fuzz.trimf(tau.universe, [13, 20, 27])
    tau['A'] = fuzz.trimf(tau.universe, [20, 27, 33.0])
    tau['MA'] = fuzz.trimf(tau.universe, [27, 33.0, 40])

    tau.view()

    # Calcular o Custo de cada individuo da população inicial
    print('\nCalcular custo:')
    for i in range(0, tamanho_populacao):
        pop_erro[i].custo, x1_temp, x2_temp, tempo = f_controle_fuzzy_pyragas(Parametros_Duffing,
                                                                                    pop_erro[i].var_fuzzy,
                                                                                    erro_dot,
                                                                                    pop_ganho[i].var_fuzzy,
                                                                                    tau)
        pop_ganho[i].custo = pop_erro[i].custo

        print('ind{}: custo: {}'.format(i,pop_erro[i].custo))
        #f_plotar_grafico(x1_temp, x2_temp, tempo, 'Custo da Pop Inicial. Individuo :{}'.format(i))

        # selecionar melhor custo da população inicial se não houver overflow
        if pop_erro[i].custo < str_Melhor_Solucao_erro.custo and pop_erro[i].custo >= 0:
            str_Melhor_Solucao_erro = pop_erro[i].deepcopy()
            str_Melhor_Solucao_ganho = pop_ganho[i].deepcopy()

            x1, x2 = x1_temp, x2_temp

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
    melhor_custo = np.empty(num_geracoes + 1)
    melhor_custo[0] = str_Melhor_Solucao_erro.custo
    str_Melhor_Solucao_erro.var_fuzzy.view()
    str_Melhor_Solucao_ganho.var_fuzzy.view()
    f_plotar_grafico(x1, x2, tempo, 'Resultado do controle do melhor indivíduo da população inicial')
    print('\nPopulação inicial:')
    print('erro: ', pop_erro)
    print('ganho: ', pop_ganho)

    #_________________________Main Loop - Iteração das gerações ________________________________________________________________
    print('\n___ Loop de iteração para evolução das gerações ___\n')

    for it in range(1, num_geracoes+1, 1):
        print('\nGeração: ', it)

        print('\nOperação de Seleção\n')
        custo_total_inverso_erro = sum(1/ind.custo for ind in pop_erro)                     # Somatório do inverso do custo para resolver problema de
                                                                                            # minimização (menor custo deve ter maior probabilidade de ser selcionado)
        prob_individuo_erro = [(1/ind.custo)/custo_total_inverso_erro for ind in pop_erro]  # probabilidade de cada infividuo ser selecionado na roleta

        pop_filhos_erro = []        # população de filhos gerados para o erro
        pop_filhos_ganho = []       # população de filhos gerados para o ganho

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
            filho1_erro, filho2_erro = GA.f_cruzamento(pai1_erro, pai2_erro, gamma_cruzamento)

            # cruzamento da filhos ganho
            filho1_ganho, filho2_ganho = GA.f_cruzamento(pai1_ganho, pai2_ganho, gamma_cruzamento)

            print('\nOperação de Mutação')

            # Operação de Mutação
            print('Operação de Mutação')
            # mutação da filhos erro
            filho1_erro.posicao = GA.f_mutacao(filho1_erro, taxa_mutacao, simga_mutacao)
            filho2_erro.posicao = GA.f_mutacao(filho2_erro, taxa_mutacao, simga_mutacao)

            # mutação da filhos ganho
            filho1_ganho.posicao = GA.f_mutacao(filho1_ganho, taxa_mutacao, simga_mutacao)
            filho2_ganho.posicao = GA.f_mutacao(filho2_ganho, taxa_mutacao, simga_mutacao)

            # Atualizar variavel fuzzy dos filhos
            filho1_erro.var_fuzzy = GA.f_atualizar_var_fuzzy('erro', UD_erro, filho1_erro.posicao)
            filho2_erro.var_fuzzy = GA.f_atualizar_var_fuzzy('erro', UD_erro, filho2_erro.posicao)

            filho1_ganho.var_fuzzy = GA.f_atualizar_var_fuzzy('ganho', UD_ganho, filho1_ganho.posicao)
            filho2_ganho.var_fuzzy = GA.f_atualizar_var_fuzzy('ganho', UD_ganho, filho2_ganho.posicao)

            #print('filho 1: ', filho1)
            #filho1.var_fuzzy.view()

            # Avaliar Custo dos filhos
            # Filho 1
            filho1_erro.custo, x1_temp, x2_temp, tempo = f_controle_fuzzy_pyragas (Parametros_Duffing,
                                                                                    filho1_erro.var_fuzzy,
                                                                                    erro_dot,
                                                                                    filho1_ganho.var_fuzzy,
                                                                                    tau)

            filho1_ganho.custo = filho1_erro.custo
            #print('Filho1. Custo: ', filho1_erro.custo)

            # selecionar melhor custo da população inicial se não houver overflow
            if filho1_erro.custo < str_Melhor_Solucao_erro.custo and filho1_erro.custo >= 0:
                str_Melhor_Solucao_erro = filho1_erro.deepcopy()
                str_Melhor_Solucao_ganho = filho1_ganho.deepcopy()

                x1, x2 = x1_temp, x2_temp

            # Filho 2
            filho2_erro.custo, x1_temp, x2_temp, tempo = f_controle_fuzzy_pyragas(Parametros_Duffing,
                                                                         filho2_erro.var_fuzzy,
                                                                         erro_dot,
                                                                         filho2_ganho.var_fuzzy,
                                                                         tau)

            filho2_ganho.custo = filho2_erro.custo

            #print('Filho2. Custo: ', filho2_erro.custo)

            # selecionar melhor custo da população inicial se não houver overflow
            if filho2_erro.custo < str_Melhor_Solucao_erro.custo and filho2_erro.custo >= 0:
                str_Melhor_Solucao_erro = filho2_erro.deepcopy()
                str_Melhor_Solucao_ganho = filho1_ganho.deepcopy()

                x1, x2 = x1_temp, x2_temp

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

        # Armazena o melhor custo da geração
        melhor_custo[it] = str_Melhor_Solucao_erro.custo
        #f_plotar_grafico(x1, x2, tempo, 'Melhor custo da geração {}'.format(it))

        print('\nPopulação da geração:', it)
        print('pop_erro: ', pop_erro)
        print('pop_ganho: ', pop_ganho)

        # Imprime melhor custo da geração
        print('\nGeração {}: Melhor custo = {}\n'.format(it, melhor_custo[it]))

    f_plotar_grafico(x1, x2, tempo, 'Melhor custo do algoritmo')

    str_Melhor_Solucao_erro.var_fuzzy.view()
    str_Melhor_Solucao_ganho.var_fuzzy.view()

    out = structure()
    out.x1 = x1
    out.y = x2
    out.melhor_solucao_erro = str_Melhor_Solucao_erro
    out.desempenho_AG = melhor_custo
    return out

# _____________________________________________________________________________________________________________________________________________
def f_plotar_grafico(p_x1, p_x2, p_tempo, titulo):
    # Malha fechada
    plt.plot(p_tempo, p_x1, color='r', lw=2, label='x');
    plt.plot(p_tempo, p_x2, color='b', lw=2, label='y');
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.title(titulo)
    plt.legend()
    plt.show()

# _____________________________________________________________________________________________________________________________________________
# Definição do Problem - Parametros do sistema de Rossler e do controlador de Pyragas

str_Parametros_Duffing = structure()             # estrutura com parametros do sistema
str_Parametros_Duffing.num_iteracoes = 4000                 # número de iterações
str_Parametros_Duffing.T = 0.01                  # tempo de amostragem
str_Parametros_Duffing.tempo = np.zeros(str_Parametros_Duffing.num_iteracoes)


# variáveis de estado
str_Parametros_Duffing.x1 = np.zeros(str_Parametros_Duffing.num_iteracoes)
str_Parametros_Duffing.x2 = np.zeros(str_Parametros_Duffing.num_iteracoes)
str_Parametros_Duffing.y = np.zeros(str_Parametros_Duffing.num_iteracoes)

# Ação de controle
str_Parametros_Duffing.u = np.zeros(str_Parametros_Duffing.num_iteracoes)

# constantes da equação de lorenz
str_Parametros_Duffing.alpha = -1
str_Parametros_Duffing.beta = 1
str_Parametros_Duffing.omega = 1.2
str_Parametros_Duffing.delta = 0.3
str_Parametros_Duffing.gama = 0.5

# Condições Iniciais
str_Parametros_Duffing.x1[0] = 2
str_Parametros_Duffing.x2[0] = 0
# _____________________________________________________________________________________________________________________________________________
# Variavies da lógica fuzzy
str_Parametros_Fuzzy = structure()

# Universo de discurso
str_Parametros_Fuzzy.UD_erro = np.arange(-10.0,10.0, str_Parametros_Duffing.T)
str_Parametros_Fuzzy.UD_erro_dot = np.arange(-0.5,0.5, str_Parametros_Duffing.T)
str_Parametros_Fuzzy.UD_saida_ganho = np.arange(0, 35.0, str_Parametros_Duffing.T)
str_Parametros_Fuzzy.UD_saida_tau = np.arange(1, 40, str_Parametros_Duffing.T)

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
str_Parametros_GA.num_geracoes = 50            # numero de gerações
str_Parametros_GA.taxa_crossover = 0.90         # probabilidade de dois indivíduos realizarem a operação de crossover
str_Parametros_GA.taxa_selecao = 0.5            # taxa de seleção da população que será mantida
str_Parametros_GA.sigma_mutacao = 0.1           # parametro usado para operação de mutação
str_Parametros_GA.gamma_cruzamento = 0.3           # parametro usado para operação de cruzamento


# _____________________________________________________________________________________________________________________________________________
# Executar GA
print('___ INÍCIO DE EXECUÇÃO ___\n\n')
print('Simulação do Sistema de Rossler')
print('Autor = Jhonatan Luiz Souza Siqueira\n\n')


out = run(str_Parametros_Duffing, str_Parametros_Fuzzy, str_Parametros_GA)

# Simulação em Malha Aberta
[x1_ma, x2_ma] = f_malha_aberta(str_Parametros_Duffing)

# Controle de pyragas
[x1pyragas, x2pyragas] = f_controle_pyragas(str_Parametros_Duffing)

# _____________________________________________________________________________________________________________________________________________
# Resultados
plt.plot(str_Parametros_Duffing.tempo, x1_ma, color='b', lw=2, label='x1');
plt.plot(str_Parametros_Duffing.tempo, x2_ma, color='r', lw=2, label='x2');
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Simulação em Malha Aberta')
plt.legend()
plt.show()

# comparação controle
plt.plot(str_Parametros_Duffing.tempo, out.y, color='r', lw=2, label='controle fuzzy-pyragas');
plt.plot(str_Parametros_Duffing.tempo, x2pyragas, color='b', lw=2, label='controle Pyragas');
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Comparação do controle')
plt.legend()
plt.show()

# comparação controle
plt.plot(str_Parametros_Duffing.tempo, out.x1, color='r', lw=2, label='controle fuzzy-pyragas');
plt.plot(str_Parametros_Duffing.tempo, x1pyragas, color='b', lw=2, label='controle Pyragas');
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Comparação do controle')
plt.legend()
plt.show()

# Malha aberta vs Malha fechada
plt.plot(str_Parametros_Duffing.tempo, x2_ma, color='r', lw=2, label='x2');
plt.plot(str_Parametros_Duffing.tempo, out.y, color='b', lw=2, label='controle fuzzy-pyragas');
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Simulação em Malha Aberta x Malha Fechada')
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

