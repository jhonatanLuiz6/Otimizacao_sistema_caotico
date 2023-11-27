# Código para simular o controle do sistema de Lorenz utiliando o controlador de Pyragas, lógica Fuzzy e GA

# ***************************************************************************************************************/
# Eng. Controle e Automação */
# Autor: Jhonatan Luiz Souza Siqueira */
# Centro Federal de Educação e Tecnologia (CEFET-MG) */
# Monografia de Graduação */
# OTIMIZAÇÃO DO MÉTODO DE PYRAGAS PARA CONTROLE DE SISTEMAS CAÓTICOS UTILIZANDO LÓGICA FUZZY E ALGORITMO GENÉTICO */
# Orientador: Marlon José do Carmo */
# **
# Algoritmo de otimização do conjunto fuzzy utiliziado no controle de sistema caótico
# Simulação para o Sistema de Lorenz
# # /*(AG Fuzzy-Pyragas)*/
# **************************************************************************************************************/
import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import cls_algoritmo_genetico as GA
import random as rd

# _____________________________________________________________________________________________________________________________________________
# Simulação do sistema em malha aberta
def f_malha_aberta(Parametros_Lorenz):
    n = Parametros_Lorenz.num_iteracoes  # número de iterações
    T = Parametros_Lorenz.T   # tempo de amostragem
    tempo_ma = Parametros_Lorenz.tempo

    # variáveis de estado
    xma = np.zeros(n)
    yma = np.zeros(n)
    zma = np.zeros(n)

    # Condições Iniciais
    xma[0] = Parametros_Lorenz.x[0]
    yma[0] = Parametros_Lorenz.y[0]
    zma[0] = Parametros_Lorenz.z[0]

    # constantes da equação de lorenz
    s = Parametros_Lorenz.sigma
    b = Parametros_Lorenz.beta
    p = Parametros_Lorenz.rho

    # Simulação
    for k in range(1, n, 1):
        xma[k] = (-s * T + 1) * xma[k - 1] + (s * T) * yma[k - 1]
        yma[k] = p * T * xma[k - 1] + (-T + 1) * yma[k - 1] - T * xma[k - 1] * zma[k - 1]
        zma[k] = T * xma[k - 1] * yma[k - 1] + (-b * T + 1) * zma[k - 1]
        tempo_ma[k] = tempo_ma[k - 1] + T

    return [xma, yma, zma, tempo_ma]

# ---------------------------------------------------------------------------------------------------------------------------------------------
# simulação do controle por Pyragas
def f_controle_pyragas(Parametros_Lorenz):
    n = Parametros_Lorenz.num_iteracoes  # número de iterações
    T = Parametros_Lorenz.T  # tempo de amostragem
    tempo_mf = Parametros_Lorenz.tempo

    # variáveis de estado
    x_py = np.zeros(n)
    y_py = np.zeros(n)
    z_py = np.zeros(n)

    u_py = np.zeros(n)

    # Condições Iniciais
    x_py[0] = Parametros_Lorenz.x[0]
    y_py[0] = Parametros_Lorenz.y[0]
    z_py[0] = Parametros_Lorenz.z[0]

    # constantes da equação de lorenz
    s = Parametros_Lorenz.sigma
    b = Parametros_Lorenz.beta
    p = Parametros_Lorenz.rho

    # parâmetros do controle Pyragas
    atraso = 100
    Tau = 5
    K = 20

    # evolução do sistema
    for k in range(1, n, 1):
        x_py [k] = (-s * T + 1) * x_py [k - 1] + (s * T) * y_py [k - 1]
        y_py [k] = p * T * x_py [k - 1] + (-T + 1) * y_py [k - 1] - T * x_py [k - 1] * z_py [k - 1] + T * u_py [k - 1]
        z_py [k] = T * x_py [k - 1] * y_py [k - 1] + (-b * T + 1) * z_py [k - 1]

        # control action
        if k > atraso:
            u_py[k] = K * (y_py[k - Tau] - y_py[k])

        tempo_mf[k] = tempo_mf[k - 1] + T

    return [x_py, y_py, z_py, tempo_mf]

# _____________________________________________________________________________________________________________________________________________
def f_controle_fuzzy_pyragas(Parametros_Lorenz, p_erro, p_erro_dot, p_ganho, p_tau):
    # Código que pode gerar os warnings como overflow devido o universo de discurso utilizado nas entradas ou saídas do controlador fuzzy
    try:
        # _________________________Informações do sistema___________________________________________________________________

        n = Parametros_Lorenz.num_iteracoes  # número de iterações
        T = Parametros_Lorenz.T  # tempo de amostragem
        tempo = Parametros_Lorenz.tempo  # array tempo

        # variáveis de estado
        xmf = np.zeros(n)
        ymf = np.zeros(n)
        zmf = np.zeros(n)

        # Condições Iniciais
        xmf[0] = Parametros_Lorenz.x[0]
        ymf[0] = Parametros_Lorenz.y[0]
        zmf[0] = Parametros_Lorenz.z[0]

        # Ação de controle
        u = Parametros_Lorenz.u
        var_erro = np.zeros(n)

        # constantes da equação de lorenz
        s = Parametros_Lorenz.sigma
        b = Parametros_Lorenz.beta
        p = Parametros_Lorenz.rho

        # Entradas do controlador Fuzzy
        erro_crisp = np.zeros(n)
        erro_dot_crisp = np.zeros(n)

        # control parameters
        atraso = 100
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
            xmf[k] = (-s * T + 1) * xmf[k - 1] + (s * T) * ymf[k - 1]
            ymf[k] = p * T * xmf[k - 1] + (-T + 1) * ymf[k - 1] - T * xmf[k - 1] * zmf[k - 1] + T * u[k - 1]
            zmf[k] = T * xmf[k - 1] * ymf[k - 1] + (-b * T + 1) * zmf[k - 1]

            # Ação de controle
            if k > atraso:
                erro_crisp[k] = ymf[k - Tau] - ymf[k]
                erro_dot_crisp[k] = (erro_crisp[k] - erro_crisp[k - 1]) * 0.1

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

        return f_custo, xmf, ymf, zmf, tempo
    except RuntimeWarning as warning:
        # Bloco executado se um RuntimeWarning for gerado
        print('[WARNING] Aviso de tempo de execução: ',  str(warning))
        return -1, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    except:
        # Bloco executado se ocorrer qualquer outra exceção não tratada
        print("[WARNING] Ocorreu um erro inesperado na simulação da população.")
        return -1, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
# ---------------------------------------------------------------------------------------------------------------------------------------------
# Função de execução do algoritimo genético
def f_run(Parametros_Lorenz, Parametros_Fuzzy, Parametros_GA):
    # _________________________Informações do sistema___________________________________________________________________
    n           = Parametros_Lorenz.num_iteracoes     # número de iterações
    T           = Parametros_Lorenz.T                 # tempo de amostragem
    tempo       = Parametros_Lorenz.tempo           # array tempo

    # variáveis de estado
    x = Parametros_Lorenz.x
    y = Parametros_Lorenz.y
    z = Parametros_Lorenz.z

    # Ação de controle
    u = np.zeros(n)

    # constantes da equação de lorenz
    s = Parametros_Lorenz.sigma
    b = Parametros_Lorenz.beta
    p = Parametros_Lorenz.rho

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
    sigma = Parametros_GA.sigma_mutacao  # parametro para operação de mutação
    gamma = Parametros_GA.gamma_cruzamento  # parametro para operação de cruzamento

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
    x_temp = np.zeros(n)
    y_temp = np.zeros(n)
    z_temp = np.zeros(n)
    tempo_temp = np.zeros(n)

    print('\n[Parâmetros do sistema]\n')
    print('sigma = {}; beta = {}; rho = {}'.format(s, b, p))
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
    erro_dot['MN'] = fuzz.trapmf(erro_dot.universe, [-6, -5, -4, -3])
    erro_dot['N'] = fuzz.trimf(erro_dot.universe, [-4, -2, 0])
    erro_dot['Z'] = fuzz.trimf(erro_dot.universe, [-2, 0, 2])
    erro_dot['P'] = fuzz.trimf(erro_dot.universe, [0, 2, 4])
    erro_dot['MP'] = fuzz.trapmf(erro_dot.universe, [2, 4, 5, 6])

    erro_dot.view()

    # criação da população da variável Fuzzy ganho
    print('\nPopulação inicial do ganho:')
    for i in range(0, tamanho_populacao):
        pts_MB, pts_B, pts_M, pts_A, pts_MA = GA.f_criar_funcao_pertinencia2(UD_ganho)

        metodo_defuzz = rd.choice(["centroid", "mom"])
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
    metodo_defuzz = "centroid"
    tau = ctrl.Consequent(UD_tau, 'tau', metodo_defuzz)

    tau['MB'] = fuzz.trimf(tau.universe, [-3.25, 5, 13.25])
    tau['B'] = fuzz.trimf(tau.universe, [1.75, 10, 18.25])
    tau['M'] = fuzz.trimf(tau.universe, [6.75, 15, 23.25])
    tau['A'] = fuzz.trimf(tau.universe, [11.75, 20, 28.25])
    tau['MA'] = fuzz.trimf(tau.universe, [16.75, 25, 33.25])

    tau.view()

    # Calcular o Custo de cada individuo da população inicial
    print('\nCalcular custo:')
    for i in range(0, tamanho_populacao):
        pop_erro[i].custo, x_temp, y_temp, z_temp, tempo_temp = f_controle_fuzzy_pyragas(Parametros_Lorenz,
                                                                                    pop_erro[i].var_fuzzy,
                                                                                    erro_dot,
                                                                                    pop_ganho[i].var_fuzzy,
                                                                                    tau)

        pop_ganho[i].custo = pop_erro[i].custo

        print('ind{}: custo: {}'.format(i,pop_erro[i].custo))
        #f_plotar_grafico(x_temp, y_temp, z_temp, tempo_temp, 'Custo da Pop Inicial. Individuo :{}'.format(i))

        # selecionar melhor custo da população inicial se não houver overflow
        if pop_erro[i].custo < str_Melhor_Solucao_erro.custo and pop_erro[i].custo >= 0:
            str_Melhor_Solucao_erro = pop_erro[i].deepcopy()
            str_Melhor_Solucao_ganho = pop_ganho[i].deepcopy()

            x, y, z, tempo = x_temp, y_temp, z_temp, tempo_temp

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
    print('população =', pop_erro)
    melhor_custo = np.empty(num_geracoes+1)
    melhor_custo[0] = str_Melhor_Solucao_erro.custo
    str_Melhor_Solucao_erro.var_fuzzy.view()
    str_Melhor_Solucao_ganho.var_fuzzy.view()
    f_plotar_grafico(x, y, z, tempo, 'Resultado do controle do melhor indivíduo da população inicial')

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

            # Se for selecionado o mesmo indivíduo, deve girar a roleta novamente
            while (indice_selecao_pai1 == indice_selecao_pai2):
                indice_selecao_pai2 = GA.f_selecao_roleta(prob_individuo_erro)

            # pais da poúlação erro
            pai1_erro = pop_erro[indice_selecao_pai1]
            pai2_erro = pop_erro[indice_selecao_pai2]

            # pais da poúlação ganho
            pai1_ganho = pop_ganho[indice_selecao_pai1]
            pai2_ganho = pop_ganho[indice_selecao_pai2]

            # Operação de Cruzamento (Crossover)
            print('\nOperação de Cruzamento')

            # cruzamento da filhos erro
            filho1_erro, filho2_erro = GA.f_cruzamento(pai1_erro, pai2_erro, 0.8)

            # cruzamento da filhos ganho
            filho1_ganho, filho2_ganho = GA.f_cruzamento(pai1_ganho, pai2_ganho, 0.3)

            # Operação de Mutação
            print('Operação de Mutação')

            # mutação da filhos erro
            filho1_erro.posicao = GA.f_mutacao(filho1_erro, taxa_mutacao, 0.1)
            filho2_erro.posicao = GA.f_mutacao(filho2_erro, taxa_mutacao, 0.1)

            # mutação da filhos ganho
            filho1_ganho.posicao = GA.f_mutacao(filho1_ganho, taxa_mutacao, 0.1)
            filho2_ganho.posicao = GA.f_mutacao(filho2_ganho, taxa_mutacao, 0.1)

            # Atualizar variavel fuzzy dos filhos
            filho1_erro.var_fuzzy = GA.f_atualizar_var_fuzzy('erro', UD_erro, filho1_erro.posicao)
            filho2_erro.var_fuzzy = GA.f_atualizar_var_fuzzy('erro', UD_erro, filho2_erro.posicao)

            filho1_ganho.var_fuzzy = GA.f_atualizar_var_fuzzy('ganho', UD_ganho, filho1_ganho.posicao)
            filho2_ganho.var_fuzzy = GA.f_atualizar_var_fuzzy('ganho', UD_ganho, filho2_ganho.posicao)

            # Avaliar Custo dos filhos
            # Filho 1
            filho1_erro.custo, x_temp, y_temp, z_temp, temp_temp = f_controle_fuzzy_pyragas(Parametros_Lorenz,
                                                                                        filho1_erro.var_fuzzy,
                                                                                        erro_dot,
                                                                                        filho1_ganho.var_fuzzy,
                                                                                        tau)

            filho1_ganho.custo = filho1_erro.custo

            print('Filho1. Custo: ', filho1_ganho.custo)

            # selecionar melhor custo da população inicial se não houver overflow
            if filho1_erro.custo < str_Melhor_Solucao_erro.custo and filho1_erro.custo >= 0:
                str_Melhor_Solucao_erro = filho1_erro.deepcopy()
                str_Melhor_Solucao_ganho = filho1_ganho.deepcopy()

                x, y, z, tempo = x_temp, y_temp, z_temp, tempo_temp

            #f_plotar_grafico(x_temp, y_temp, z_temp, tempo, 'Resultado do  cruzamento (filho1)')

            # Filho 2
            #print('filho 2: ', filho2)
            #filho2.var_fuzzy.view()

            filho2_erro.custo, x_temp, y_temp, z_temp, tempo_temp = f_controle_fuzzy_pyragas(Parametros_Lorenz,
                                                                                   filho2_erro.var_fuzzy,
                                                                                   erro_dot,
                                                                                   filho2_ganho.var_fuzzy,
                                                                                   tau)

            filho2_ganho.custo = filho2_erro.custo

            print('Filho2. Custo: ', filho2_ganho.custo)

            # selecionar melhor custo da população inicial se não houver overflow
            if filho2_erro.custo < str_Melhor_Solucao_erro.custo and filho2_erro.custo >= 0:
                str_Melhor_Solucao_erro = filho2_erro.deepcopy()
                str_Melhor_Solucao_ganho = filho2_ganho.deepcopy()
                str_Melhor_Solucao_tau = filho2_ganho.deepcopy()

                x, y, z, tempo = x_temp, y_temp, z_temp, tempo_temp
            #f_plotar_grafico(x_temp, y_temp, z_temp, tempo, 'Resultado do  cruzamento (filho2)')

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
    out.y = y
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
# Definição do Problema - Parametros do sistema de Lorenz

str_Parametros_Lorenz = structure()             # estrutura com parametros do sistema
str_Parametros_Lorenz.num_iteracoes = 1000      # número de iterações
str_Parametros_Lorenz.T = 0.01                  # tempo de amostragem
str_Parametros_Lorenz.tempo = np.zeros(str_Parametros_Lorenz.num_iteracoes)

# variáveis de estado
str_Parametros_Lorenz.x = np.zeros(str_Parametros_Lorenz.num_iteracoes)
str_Parametros_Lorenz.y = np.zeros(str_Parametros_Lorenz.num_iteracoes)
str_Parametros_Lorenz.z = np.zeros(str_Parametros_Lorenz.num_iteracoes)

# Ação de controle
str_Parametros_Lorenz.u = np.zeros(str_Parametros_Lorenz.num_iteracoes)

# constantes da equação de lorenz
str_Parametros_Lorenz.sigma = 10
str_Parametros_Lorenz.beta = 8/3
str_Parametros_Lorenz.rho = 28

# Condições Iniciais
str_Parametros_Lorenz.x[0] = 0.1
str_Parametros_Lorenz.y[0] = 0.2
str_Parametros_Lorenz.z[0] = 0.3

# _____________________________________________________________________________________________________________________________________________
# Variavies da lógica fuzzy

str_Parametros_Fuzzy = structure()
str_regras = structure()

# Universo de discurso
str_Parametros_Fuzzy.UD_erro = np.arange(-35.0,35.0, str_Parametros_Lorenz.T)
str_Parametros_Fuzzy.UD_erro_dot = np.arange(-5.0,5.0, str_Parametros_Lorenz.T)
str_Parametros_Fuzzy.UD_saida_ganho = np.arange(-30,30.0, str_Parametros_Lorenz.T)
str_Parametros_Fuzzy.UD_saida_tau = np.arange(1,30.0, str_Parametros_Lorenz.T)     # ud usado qnd não otmizar o ganho
#str_Parametros_Fuzzy.UD_saida_tau = np.arange(1,25.0, str_Parametros_Lorenz.T)

# Entrada do Controlador Fuzzy
str_Parametros_Fuzzy.erro = ctrl.Antecedent(str_Parametros_Fuzzy.UD_erro, 'erro')
str_Parametros_Fuzzy.erro_dot = ctrl.Antecedent(str_Parametros_Fuzzy.UD_derro, 'erro_dot')

# Saida do Controlador
str_Parametros_Fuzzy.saida_ganho = ctrl.Consequent(str_Parametros_Fuzzy.UD_saida_ganho, 'saida_ganho')
str_Parametros_Fuzzy.saida_tau = ctrl.Consequent(str_Parametros_Fuzzy.UD_saida_tau, 'saida_tau')

# _____________________________________________________________________________________________________________________________________________
# Parâmetros do Algoritmo Genético

str_Parametros_GA = structure()
str_Parametros_GA.tamanho_populacao = 8         # Tamanho da população
str_Parametros_GA.taxa_mutacao = 0.10           # Taxa de Mutação. Geralemente utiliza-se (100 - taxa_crossover)
str_Parametros_GA.num_geracoes = 100              # numero de gerações
str_Parametros_GA.taxa_crossover = 0.90         # probabilidade de dois indivíduos realizarem a operação de crossover
str_Parametros_GA.taxa_selecao = 0.5            # taxa de seleção da população que será mantida
str_Parametros_GA.sigma_mutacao = 0.1           # parametro usado para operação de mutação
str_Parametros_GA.gamma_cruzamento = 0.5           # parametro usado para operação de cruzamento

# _____________________________________________________________________________________________________________________________________________
# Execução do GA
print('___ INÍCIO DE EXECUÇÃO ___\n\n')
print('Simulação do Sistema de Lorenz')
print('Autor = Jhonatan Luiz Souza Siqueira\n\n')

out = f_run(str_Parametros_Lorenz, str_Parametros_Fuzzy, str_Parametros_GA)

# Simulação em Malha Aberta
[x_ma, y_ma, z_ma, tempo_ma] = f_malha_aberta(str_Parametros_Lorenz)

# Simulação do controle de pyragas
[xp, yp, zp, tempo_mf] = f_controle_pyragas(str_Parametros_Lorenz)

# _____________________________________________________________________________________________________________________________________________
# Resultados
# Malha aberta
plt.plot(tempo_ma, x_ma, color='r', lw=2, label='x');
plt.plot(tempo_ma, y_ma, color='b', lw=2, label='y');
plt.plot(tempo_ma, z_ma, color='g', lw=2, label='z');
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Simulação em Malha Aberta')
plt.legend()
plt.show()

# Comparação com malha fechada
plt.plot(tempo_ma, y_ma, color='r', lw=2, label='malha aberta');
plt.plot(str_Parametros_Lorenz.tempo, out.y, color='b', lw=2, label='controle fuzzy-pyragas');
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Malha aberta x Malha fechada')
plt.legend()
plt.show()

# Comparação Pyragas vs Fuzzy-Pyragas
plt.plot(tempo_mf, yp, color='r', lw=2, label='controle Pyragas');
plt.plot(str_Parametros_Lorenz.tempo, out.y, color='b', lw=2, label='controle Fuzzy-Pyragas');
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