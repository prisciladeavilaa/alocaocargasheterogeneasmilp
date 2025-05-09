# Versão que considera as UM individualmente

import gurobipy as gp
from gurobipy import GRB

import random
# Manipulação e análise de dados em formato tabular - DataFrames.
import pandas as pd
# utilizar PANDAS para ler/processar os arquivos CSV de forma mais avançada, analisar os resultados da otimização e gerar relatórios

import csv  # Leitura e escrita de arquivos no formato CSV
# Cria dicionários com valores padrão para chaves não existentes (destino_para_clientes = defaultdict(list))
from collections import defaultdict
import os  # Para operações com caminhos de arquivo

from datetime import datetime

import os
# os.environ['GRB_LICENSE_ID'] = '5afd4a9d-d0cd-4de6-a329-20ed5d23f84b' #configura uma variável de ambiente que o Gurobi verificará automaticamente

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import numpy as np

TIMEOUT = 60  # em segundos
# ====================== CARREGAMENTO DE DADOS ====================== #

# Lê o arquivo csv e organiza os dados em estruturas Python
# Retorna um dicionário com parametros, veiculos, ums e clientes


def carregar_dados(caminho_arquivo):

    dados = {
        'parametros': {},  # Dados globais (ex penalidades)
        'veiculos': [],   # Lista de veículos disponíveis
        'ums': [],        # Unidades Metalicas para transportar
        'clientes': []    # Clientes por região
    }

    # Abre o arquivo CSV no modo leitura ('r') com encoding UTF-8
    with open(caminho_arquivo, mode='r', encoding='utf-8') as file:
        # DictReader lê cada linha como um dicionário
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:  # para cada linha do arquivo
            tipo = row['tipo']

            if tipo == 'parametro':  # Ex: Tipo: Penalidade por não alocação | Valor: 0.1
                dados['parametros'][row['descricao']] = float(row['valor'])

            elif tipo == 'cliente':
                dados['clientes'].append({
                    'id': int(row['id']),
                    'nome': row['descricao'],
                    'destino': row['destino']
                })

            elif tipo == 'veiculo':
                dados['veiculos'].append({
                    'id': int(row['id']),
                    'tipo': row['descricao'].replace('Veiculo_', ''),
                    'capacidade_peso': float(row['capacidade_peso']),
                    'capacidade_volume': float(row['capacidade_vol']),
                    'custo': float(row['custo']),
                    'carga_minima': float(row['carga_minima']),
                    'destino': row['destino'] if 'destino' in row else None
                })

            elif tipo == 'um':
                cliente_id = int(row['cliente'])
                # Se o cliente for encontrado, pega o campo 'destino'. Se não for encontrado, define um valor padrão (vazio)
                # Encontra o cliente correspondente na lista de clientes
                destino = next(
                    (c['destino'] for c in dados['clientes'] if c['id'] == cliente_id), '')

                compatibilidade = row['compatibilidade'].strip()
                if not compatibilidade:
                    compatibilidade = ",".join(
                        str(v['tipo']) for v in dados['veiculos'])

                dados['ums'].append({
                    'id': int(row['id']),
                    'tipo': row['descricao'],
                    'peso': float(row['peso']),
                    'volume': float(row['volume']),
                    'destino': destino,
                    'cliente': cliente_id,
                    'compatibilidade': row['compatibilidade'] or ",".join(str(v['tipo']) for v in dados['veiculos']),
                    'restricao': row['restricao'],
                    'penalidade': float(row['penalidade'])
                })

    # #Verifica se os dados foram carregados corretamente
    # print(f"{'='*80}")
    # print(f"Dados carregados: Clientes: {dados['clientes']}")
    # print(f"Veículos: {dados['veiculos']}")
    # print(f"Ums: {dados['ums']}")
    # for um in dados['ums']:
    #     print(f"Um: {um}")
    #     for k, v in um.items():
    #         print(f"{k}: {v}")
    #         print(f"Compatibilidade " v.split(','))
    # print(f"Parâmetros: {dados['parametros']}")
    # print(f"{'='*80}")

    return dados


def criar_instancia(tipo_instancia):
    # Armazena um dicionário com os dados na variável dados
    dados = carregar_dados(tipo_instancia)

    return {  # Retorna os componentes do dicionário
        "veiculos": dados['veiculos'],
        "ums": dados['ums'],
        "clientes": dados['clientes'],
        # "penalidade": dados['parametros']['Penalidade por não alocação']
    }

# ====================== MODELO GUROBI ====================== #


def criar_modelo(instancia):
    # Cria um novo modelo de otimização no Gurobi chamado "AlocacaoCargas"
    model = gp.Model("AlocacaoCargas")

    # Extrai os dados da instância
    veiculos = instancia["veiculos"]
    ums = instancia["ums"]
    clientes = instancia["clientes"]
    # penalidade = instancia["penalidade"]

    # Cria um dicionário que relaciona cada região aos clientes dessa região {"Região_1": ["Cliente_R1_1", "Cliente_R1_2"], ...}
    destino_para_clientes = defaultdict(list)
    for cliente in clientes:
        destino_para_clientes[cliente['destino']].append(cliente['nome'])

    # Parâmetro delta_{cr}: 1 se cliente c pertence à região r (veículo com destino r pode atender c)
    # Matriz de compatibilidade veículo->cliente

    # Matriz que indica se um veículo v pode atender um cliente c, ex.: delta[(1, 2)] = 1 significa que o cliente 1 pode ser atendido pelo veículo 2
    delta = {}
    for v in veiculos:
        for c in clientes:
            # 1 se o veículo e cliente têm o mesmo destino, 0 c.c.
            delta[(c['id'], v['id'])] = 1 if v['destino'] == c['destino'] else 0

    # print("=================================================================")
    # print("Clientes:", clientes)
    # print("Primeiro veículo:", veiculos[0])
    # print("Primeira UM:", ums[0])
    # print("Matriz delta:", delta)
    # print("=================================================================")

    # Variáveis de decisão
    # x[i_id, v_id, c_id] - Binária (1 se UM i for alocada ao veículo v para cliente c)
    x = {}
    y = {}  # y[v_id, c_id] - Binária (1 se veículo v for usado para cliente c)
    alpha = {}  # alpha[v_id] - Binária (1 se veículo v estiver ativo)
    # z[v_id] - Contínua (valor do frete morto para veículo v - custo adicional se o veículo não atingir a carga mínima)
    z = {}

    # Cria as variáveis binárias x no modelo Gurobi, uma para cada combinação de carga (i), veículo (v) e cliente (c)
    for i in ums:
        for v in veiculos:
            for c in clientes:
                x[(i["id"], v["id"], c["id"])] = model.addVar(vtype=GRB.BINARY,
                                                              # cria as x[i,v,c]
                                                              name=f"x_{i['id']}_{v['id']}_{c['id']}")

    # Cria as tres variáveis de decisão relacionadas aos veículos
    for v in veiculos:
        # Variável binária alpha que indica se o veículo v está ativo - 1 se o veículo transportar qualquer carga
        alpha[v["id"]] = model.addVar(
            vtype=GRB.BINARY, name=f"alpha_{v['id']}")
        # variável não negativa z para o frete morto - penalização por usar frete morto!
        z[v["id"]] = model.addVar(lb=0, name=f"z_{v['id']}")
        for c in clientes:
            # variável binaria y vincula veículo v ao cliente c - 1 se veículo v for usado para cliente c
            y[(v["id"], c["id"])] = model.addVar(
                vtype=GRB.BINARY, name=f"y_{v['id']}_{c['id']}")

    # ====================== FUNÇÃO OBJETIVO ====================== #

    # Calcula o custo de não alocar uma carga i:
    # Soma todas as alocações possíveis da UM i em todos os veículos v e cliente c
    # Se a carga não for alocada, (1 - soma) será 1, e o custo será peso * penalidade.
    # Se for alocada, (1 - soma) será 0 (sem custo).
    custo_nao_alocacao = gp.quicksum(
        # 1 - ... para inverter a lógica: 1 - 1 = 0 (sem custo se alocada), 1 - 0 = 1 (custo se não alocada)
        i["peso"] * i["penalidade"] * (1 - gp.quicksum(x[(i["id"], v["id"], c["id"])]
                                                       for v in veiculos for c in clientes))
        for i in ums
    )
    # Considerar penalidade individual da um
    # algumas UMs têm penalidades muito maiores independente do peso
    # a penalidade individual da UM já incorpora o peso e os outros critérios
    # custo_nao_alocacao = gp.quicksum(
    #     i["penalidade"] * (1 - gp.quicksum(x[(i["id"], v["id"], c["id"])]
    #                     for v in veiculos for c in clientes))
    #     for i in ums
    # )

    # Soma o frete morto de todos os veículos. Se o veículo transporta menos que a carga mínima, z vai ser positivo.
    # O frete morto calculado nas restrições
    custo_frete_morto = gp.quicksum(z[v["id"]] for v in veiculos)

    # Custo total dos veículos ativos: Cada veículo só é cobrado uma vez quando ativo (alpha[v["id"]] = 1)
    custo_transporte = gp.quicksum(
        v["custo"] * alpha[v["id"]]  # alpha que indica se o veículo está ativo |
        for v in veiculos

    )

    # FO do modelo calculada
    model.setObjective(custo_nao_alocacao +
                       custo_frete_morto + custo_transporte, GRB.MINIMIZE)

    # ====================== RESTRIÇÕES ====================== #

    for v in veiculos:
        # R1
        # Garante que a soma dos pesos das UMs alocadas a um veículo não exceda a capacidade.
        model.addConstr(
            gp.quicksum(i["peso"] * x[(i["id"], v["id"], c["id"])]
                        for i in ums for c in clientes) <= v["capacidade_peso"],
            name=f"cap_peso_{v['id']}"
        )
        # Garante que a soma dos volumes das UMs alocadas a um veículo não exceda a capacidade.
        model.addConstr(
            gp.quicksum(i["volume"] * x[(i["id"], v["id"], c["id"])]
                        for i in ums for c in clientes) <= v["capacidade_volume"],
            name=f"cap_vol_{v['id']}"
        )
        # R2
        # Frete morto: diferença entre carga mínima exigida e carga real transportada.
        # Se a carga real for menor que a mínima, z[v["id"]] será positivo. somente se o veículo estiver ativo (alpha = 1).
        model.addConstr(
            z[v["id"]] >= alpha[v["id"]] * v["carga_minima"] -
            gp.quicksum(i["peso"] * x[(i["id"], v["id"], c["id"])]
                        for i in ums for c in clientes),
            name=f"frete_morto_{v['id']}"
        )
        # Se um veículo v é vinculado a um cliente (y = 1), alpha = 1 (veículo ativo)
        for c in clientes:
            model.addConstr(
                alpha[v["id"]] >= y[(v["id"], c["id"])],
                name=f"ativacao_{v['id']}_{c['id']}"
            )

    for i in ums:
        # R3
        # Alocação única: Cada carga i pode ser alocada a, no máximo, um único veículo v.
        model.addConstr(
            gp.quicksum(x[(i["id"], v["id"], c["id"])]
                        for v in veiculos for c in clientes) <= 1,
            name=f"alocacao_unica_{i['id']}"
        )

        for v in veiculos:
            for c in clientes:
                # R4:
                # Remove espaços em branco no início e final de cada elemento:
                veiculos_compatíveis = [vc.strip()
                                        for vc in i['compatibilidade'].split(',')]
                # importante caso haja espaços acidentais (ex: "Veiculo_Truck, Veiculo_Bi-truck")
                gamma = 1 if v['tipo'] in veiculos_compatíveis else 0
                model.addConstr(
                    x[(i["id"], v["id"], c["id"])] <= gamma,
                    name=f"compat_{i['id']}_{v['id']}_{c['id']}"
                )
                # ERRO: Se uma UM for compatível com um veículo mas estiver destinada a um cliente diferente do que o veículo atende, ela não será alocada.
                # #R5
                # Associação alocação e uso: Garante que a alocação de cargas i seja feita em um veículo v ativo, ou seja, com pelo menos um cliente de uma região- R6 Resolve!
                model.addConstr(
                    x[(i["id"], v["id"], c["id"])] <= y[(
                        v["id"], c["id"])],  # x_ivc <= y_vc
                    name=f"aloc_uso_{i['id']}_{v['id']}_{c['id']}"
                )
                # # só aplica R5 se delta=1
                # model.addConstr(
                #     x[(i["id"], v["id"], c["id"])] <= y[(v["id"], c["id"])] * delta[(c["id"], v["id"])],
                #     name=f"aloc_uso_cond_{i['id']}_{v['id']}_{c['id']}"
                # )
                # R6: Garante que a carga i só pode ser alocada ao veículo v para o cliente c se o destino do veículo for compatível com o do cliente (delta = 1).
                model.addConstr(
                    x[(i["id"], v["id"], c["id"])] <= delta.get(
                        (c["id"], v["id"]), 0),
                    name=f"destino_{i['id']}_{v['id']}_{c['id']}"
                )
                # R7: Força pelo menos uma alocação
                # model.addConstr(
                #  gp.quicksum(x[(i["id"], v["id"], c["id"])]
                #           for i in ums for v in veiculos for c in clientes) >= 1,
                #  name="alocacao_minima"
                # )
            # Garante que alpha[v["id"]] só pode ser 1 se o veículo v for usado para pelo menos um cliente (y[(v["id"], c["id"])] = 1 para algum c
            # Completa a lógica junto com a restrição existente que força alpha a ser 1 se qualquer y for 1
            for v in veiculos:
                model.addConstr(
                    alpha[v["id"]] <= gp.quicksum(
                        y[(v["id"], c["id"])] for c in clientes),
                    name=f"ativacao_max_{v['id']}"
                )

    return model, x, y, z, alpha

# ====================== FUNÇÕES DE VISUALIZAÇÃO ====================== #


def gerar_visualizacoes(resultados, instancia, pasta_saida):

    # Gera e salva todas as visualizações para uma instância.

    #     resultados: Dicionário com os resultados da otimização
    #     instancia: Dicionário com os dados da instância
    #     pasta_saida: Caminho para salvar as imagens

    os.makedirs(pasta_saida, exist_ok=True)
    nome_base = resultados['tipo_instancia']

    # 1. Gráficos de Desempenho
    plot_tempo_execucao(resultados, pasta_saida, nome_base)
    plot_gap_otimizacao(resultados, pasta_saida, nome_base)
    plot_status_solucao(resultados, pasta_saida, nome_base)

    # 2. Alocação de Recursos
    plot_utilizacao_veiculos(resultados, pasta_saida, nome_base)
    plot_distribuicao_utilizacao(resultados, pasta_saida, nome_base)
    plot_ums_por_veiculo(resultados, pasta_saida, nome_base)

    # 3. Custos e Penalidades
    plot_composicao_custos(resultados, pasta_saida, nome_base)
    plot_custo_por_componente(resultados, pasta_saida, nome_base)
    plot_penalidades_nao_alocacao(resultados, pasta_saida, nome_base)

    # 4. UMs Não Alocadas
    if resultados['ums_nao_alocadas'] > 0:
        plot_heatmap_compatibilidade(instancia, pasta_saida, nome_base)
        plot_distribuicao_ums_nao_alocadas(
            instancia, resultados, pasta_saida, nome_base)


def plot_tempo_execucao(resultados, pasta_saida, nome_base):
    plt.figure(figsize=(10, 6))
    plt.bar(nome_base, resultados['tempo_execucao'], color='skyblue')
    plt.axhline(y=TIMEOUT, color='r', linestyle='--', label='Timeout')
    plt.ylabel('Tempo (segundos)')
    plt.title('Tempo de Execução')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_tempo_execucao.png"), dpi=300)
    plt.close()


def plot_gap_otimizacao(resultados, pasta_saida, nome_base):
    if resultados['gap_otimizacao'] is not None:
        plt.figure(figsize=(8, 5))
        plt.bar(nome_base, resultados['gap_otimizacao'], color='orange')
        plt.ylabel('GAP (%)')
        plt.title('GAP de Otimização')
        plt.tight_layout()
        plt.savefig(os.path.join(
            pasta_saida, f"{nome_base}_gap_otimizacao.png"), dpi=300)
        plt.close()


def plot_status_solucao(resultados, pasta_saida, nome_base):
    status_map = {
        GRB.OPTIMAL: "Ótimo",
        GRB.TIME_LIMIT: "Timeout",
        GRB.INFEASIBLE: "Inviável",
        GRB.INF_OR_UNBD: "Infinito/Ilimitado",
        GRB.UNBOUNDED: "Ilimitado"
    }
    status = status_map.get(resultados['status'], "Desconhecido")

    plt.figure(figsize=(6, 6))
    plt.pie([1], labels=[status], autopct='%1.0f%%', colors=['lightgreen'])
    plt.title('Status da Solução')
    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_status_solucao.png"), dpi=300)
    plt.close()


def plot_utilizacao_veiculos(resultados, pasta_saida, nome_base):
    if not resultados['alocacoes']:
        return

    df = pd.DataFrame(resultados['alocacoes'])
    df = df.sort_values('veiculo_id')

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    x = np.arange(len(df))

    bars1 = ax.bar(x - bar_width/2,
                   df['peso_total'], bar_width, label='Peso Real')
    bars2 = ax.bar(x + bar_width/2,
                   df['peso_minimo'], bar_width, label='Peso Mínimo')

    ax.set_xlabel('Veículos')
    ax.set_ylabel('Peso (kg)')
    ax.set_title('Comparação: Peso Real vs Peso Mínimo')
    ax.set_xticks(x)
    ax.set_xticklabels(df['veiculo_id'])
    ax.legend()

    for i, cap in enumerate(df['capacidade_peso']):
        ax.axhline(y=cap, xmin=(i - 0.5)/len(x), xmax=(i + 0.5)/len(x),
                   color='r', linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_utilizacao_veiculos.png"), dpi=300)
    plt.close()


def plot_distribuicao_utilizacao(resultados, pasta_saida, nome_base):
    if not resultados['alocacoes']:
        return

    df = pd.DataFrame(resultados['alocacoes'])

    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='taxa_utilizacao_peso',
                 bins=10, kde=True, color='skyblue')
    plt.xlabel('Taxa de Utilização de Peso (%)')
    plt.ylabel('Número de Veículos')
    plt.title('Distribuição das Taxas de Utilização de Peso')
    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_distribuicao_utilizacao.png"), dpi=300)
    plt.close()


def plot_ums_por_veiculo(resultados, pasta_saida, nome_base):
    if not resultados['alocacoes']:
        return

    df = pd.DataFrame(resultados['alocacoes'])
    df['num_cargas'] = df['cargas'].apply(len)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='veiculo_id', y='num_cargas',
                hue='veiculo_tipo', dodge=False)
    plt.xlabel('ID do Veículo')
    plt.ylabel('Número de UMs Transportadas')
    plt.title('Distribuição de UMs por Veículo')
    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_ums_por_veiculo.png"), dpi=300)
    plt.close()


def plot_composicao_custos(resultados, pasta_saida, nome_base):
    componentes = ['Transporte', 'Frete Morto', 'Não Alocação']
    valores = [
        resultados['custo_transporte'],
        resultados['frete_morto_total'],
        resultados['custo_nao_alocacao']
    ]

    plt.figure(figsize=(8, 8))
    plt.pie(valores, labels=componentes, autopct='%1.1f%%',
            colors=['#66b3ff', '#ff9999', '#99ff99'])
    plt.title('Composição do Custo Total')
    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_composicao_custos.png"), dpi=300)
    plt.close()


def plot_custo_por_componente(resultados, pasta_saida, nome_base):
    componentes = ['Transporte', 'Frete Morto', 'Não Alocação']
    valores = [
        resultados['custo_transporte'],
        resultados['frete_morto_total'],
        resultados['custo_nao_alocacao']
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(componentes, valores, color=['blue', 'red', 'green'])
    plt.ylabel('Custo (R$)')
    plt.title('Custo por Componente')

    # Adiciona valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'R${height:,.2f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_custo_por_componente.png"), dpi=300)
    plt.close()


def plot_penalidades_nao_alocacao(resultados, pasta_saida, nome_base):
    if resultados['ums_nao_alocadas'] == 0:
        return

    dados = {
        'Peso Não Alocado': resultados['peso_nao_alocado'],
        'Volume Não Alocado': resultados['volume_nao_alocado']
    }

    plt.figure(figsize=(10, 6))
    bars = plt.bar(dados.keys(), dados.values(), color=['orange', 'purple'])
    plt.ylabel('Valor Total')
    plt.title('Recursos Não Alocados')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:,.2f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_penalidades_nao_alocacao.png"), dpi=300)
    plt.close()


def plot_heatmap_compatibilidade(instancia, pasta_saida, nome_base):
    # Cria matriz de compatibilidade
    compat_data = []
    for um in instancia['ums']:
        compat_veiculos = []
        for veiculo in instancia['veiculos']:
            compat = 1 if veiculo['tipo'] in um['compatibilidade'].split(
                ',') else 0
            compat_veiculos.append(compat)
        compat_data.append(compat_veiculos)

    df = pd.DataFrame(
        compat_data,
        index=[f"UM_{um['id']}" for um in instancia['ums']],
        columns=[f"V_{v['id']}({v['tipo']})" for v in instancia['veiculos']]
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(df, cmap="Blues", cbar=False)
    plt.title('Matriz de Compatibilidade UMs x Veículos')
    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_heatmap_compatibilidade.png"), dpi=300)
    plt.close()


def plot_distribuicao_ums_nao_alocadas(instancia, resultados, pasta_saida, nome_base):
    alocados_ids = set()
    for aloc in resultados['alocacoes']:
        alocados_ids.update(aloc['cargas'])

    ums_nao_alocadas = [um for um in instancia['ums']
                        if um['id'] not in alocados_ids]

    if not ums_nao_alocadas:
        return

    df = pd.DataFrame(ums_nao_alocadas)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.boxplot(data=df, y='peso', ax=axes[0])
    axes[0].set_title('Distribuição de Peso das UMs Não Alocadas')

    sns.boxplot(data=df, y='volume', ax=axes[1])
    axes[1].set_title('Distribuição de Volume das UMs Não Alocadas')

    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_distribuicao_ums_nao_alocadas.png"), dpi=300)
    plt.close()

# ====================== EXECUÇÃO CONTROLADA ====================== #


def executar_instancia_com_timeout(tipo_instancia, instancia):

    try:
        print(f"\n{'='*80}")
        print(f"INICIANDO INSTÂNCIA: {tipo_instancia.upper()}")
        print(f"{'='*80}")

        # Cria o modelo
        modelo, x, y, z, alpha = criar_modelo(instancia)

        # Configura o timeout (1 hora = 3600 segundos)
        modelo.Params.TimeLimit = TIMEOUT  # timeout em segundos

        # Gera log com resultados
        modelo.Params.LogFile = os.path.join(os.path.dirname(
            __file__), 'Otimizacao', 'Resultados', f"gurobi_log_{tipo_instancia}.log")

        # MANTER VALORES PADRÃO DO GUROBI E MENCIONAR ISSO NO ARTIGO
        # Para agilizar os testes
        # modelo.Params.MIPGap = 0.01       # Aceita 1% de gap
        # modelo.Params.MIPFocus = 1        # Foco em soluções factíveis rápidas
        # modelo.Params.Heuristics = 0.5    # Aumenta esforço em heurísticas

        # Obter mais informações da otimização
        # modelo.Params.MIPGapAbs = 1e-6  # Precisão absoluta
        # modelo.Params.MIPGap = 0.0      # Busca o ótimo (gap zero)
        modelo.Params.OutputFlag = 1    # Habilita logs do solver

        # Otimiza o modelo com o solver do gurobi
        modelo.optimize()

        # Cria um dicionário para armazenar todos os resultados e inicializa com valores zerados
        resultados = {
            'tipo_instancia': tipo_instancia,
            'status': modelo.status,
            'tempo_execucao': modelo.Runtime,
            'custo_total': None,
            'veiculos_ativos': 0,
            'veiculos_inativos': len(instancia["veiculos"]),
            'ums_alocadas': 0,
            'ums_nao_alocadas': len(instancia["ums"]),
            'peso_nao_alocado': 0,
            'volume_nao_alocado': 0,
            'frete_morto_total': 0,
            'custo_transporte': 0,
            'custo_nao_alocacao': 0,
            'alocacoes': [],
            # Dados da solução:
            'tempo_para_otimo': modelo.RunTime if modelo.status == GRB.OPTIMAL else None,
            'melhor_solucao': modelo.ObjVal if modelo.SolCount > 0 else None,
            # 'solucao_relaxada': modelo.ObjBound if hasattr(modelo, 'ObjBound') else None,
            'solucao_relaxada': modelo.ObjBound if modelo.SolCount > 0 else None,
            # Em porcentagem
            'gap_otimizacao': modelo.MIPGap*100 if hasattr(modelo, 'MIPGap') else None,
        }

        if modelo.SolCount > 0:  # Se encontrar qualquer solução #modelo.status == GRB.OPTIMAL:
            # for i in instancia["ums"]:
            #     for v in instancia["veiculos"]:
            #         for c in instancia["clientes"]:
            #             chave = (i["id"], v["id"], c["id"])
            #             valor = x[chave].x  # Valor da variável na solução
            #             x_val[chave] = valor

            # x_ivc: 1 se a UM 1 foi alocada ao veículo v para o cliente c
            x_val = {(i["id"], v["id"], c["id"]): x[(i["id"], v["id"], c["id"])].x
                     # .x retorna o valor da variável na solução encontrada.
                     # Se a solução foi x = 1 (alocada), x[(...)].x retornará 1.0.
                     # Se não foi alocada, retornará 0.0
                     for i in instancia["ums"]
                     for v in instancia["veiculos"]
                     for c in instancia["clientes"]}

            # y_vc: 1 se o veículo v foi usado para atender o cliente c e 0 caso contrário
            y_val = {(v["id"], c["id"]): y[(v["id"], c["id"])].x
                     for v in instancia["veiculos"]
                     for c in instancia["clientes"]}

            # z_v: valor do frete mortopara o veículo v
            z_val = {v["id"]: z[v["id"]].x for v in instancia["veiculos"]}

            # print("\n🟠🟠🟠🟠🟠 Valores das variáveis:")
            # # Verificar se há valores 1.0
            # print("x_val samples:", list(x_val.values())[:5])
            # print("y_val samples:", list(y_val.values())[:5])
            # print("z_val samples:", list(z_val.values())[:5])

            if hasattr(modelo, 'ObjVal'):
                # Evitar erros quando os atributos não estão disponíveis
                resultados['custo_total'] = modelo.ObjVal
            # Calcula métricas
            # valor da função objetivo na solução encontrada pelo Gurobi. Armazena o custo total da solução (Custo de transporte + Frete morto + Penalidades por não alocação) no dicionário de resultados
            resultados['custo_total'] = modelo.objVal

            resultados['veiculos_ativos'] = sum(  # conta quantos veículos satisfazem a condição abaixo:
                # Para cada veículo v, verifica:
                1 for v in instancia["veiculos"]
                if any(x_val.get((i["id"], v["id"], c["id"]), 0) > 0.9  # ANY: se existe QUALQUER UM i alocada a ele (qualquer cliente c)
                       for i in instancia["ums"]
                       for c in instancia["clientes"])
                # verifica se o veículo foi marcado como ativo
                or alpha[v["id"]].x > 0.9
            )

            # número de veículos ativos subtraído do total de veículos disponíveis
            resultados['veiculos_inativos'] = len(
                instancia["veiculos"]) - resultados['veiculos_ativos']

            # Cargas não alocadas
            nao_alocadas = [  # Cria uma lista com os IDs das UMs que não foram alocadas a nenhum veículo
                i["id"] for i in instancia["ums"]  # pra cada UM i verifica:
                if all(x_val.get((i["id"], v["id"], c["id"]), 0) < 0.1  # se o valor de x_val (alocação) é menor que 0.1 (ou seja, =~ 0)
                       # Para todos os veículos v
                       for v in instancia["veiculos"]
                       # e para todos os cliente c
                       for c in instancia["clientes"])
            ]

            # Contabiliza UMs alocadas e não alocadas
            resultados['ums_nao_alocadas'] = len(nao_alocadas)
            resultados['ums_alocadas'] = len(
                instancia["ums"]) - len(nao_alocadas)

            # Soma o peso e o volume total das UMs não alocadas
            resultados['peso_nao_alocado'] = sum(
                i["peso"] for i in instancia["ums"] if i["id"] in nao_alocadas)
            resultados['volume_nao_alocado'] = sum(
                i["volume"] for i in instancia["ums"] if i["id"] in nao_alocadas)

            # Custos detalhados

            # Soma todos os valores de frete morto dos veículos. z_val: dicionário que mapeia veículo → valor do frete morto
            resultados['frete_morto_total'] = sum(z_val.values())

            # Calcula custo total do transporte - soma dos custos dos veículos ativos
            resultados['custo_transporte'] = sum(
                # custo do veículo (v["custo"]) * y_val (se o veículo foi usado para aquele cliente (retorna 1 se usado ou 0 cc))
                v["custo"] * y_val.get((v["id"], c["id"]), 0)
                # Para cada combinação veículo-cliente:
                for v in instancia["veiculos"]
                for c in instancia["clientes"]
            )

            # Calcula custo de não alocação
            resultados['custo_nao_alocacao'] = sum(  # Soma todas as penalidades aplicadas às cargas não alocadas
                i["peso"] * instancia["penalidade"] * (1 - sum(  # Aplica a penalidade: peso * penalidade * (1 - total_alocacoes)
                    # Se alocada: 1 - 1 = 0 (sem custo), se não alocada: 1 - 0 = 1 (custo máximo)
                    # x_ivc = 1 se a UM i foi alocada ao veículo v para o cliente c e 0 se não foi alocada
                    x_val.get((i["id"], v["id"], c["id"]), 0)
                    for v in instancia["veiculos"]
                    for c in instancia["clientes"]))
                for i in instancia["ums"]  # Para cada UM
            )
            # Agora calcula com penalidade individual da um
            # resultados['custo_nao_alocacao'] = sum(  # Soma todas as penalidades aplicadas às cargas não alocadas
            #     i["penalidade"] * (1 - sum( # Aplica a penalidade: penalidade * (1 - total_alocacoes)
            #         # Se alocada: 1 - 1 = 0 (sem custo), se não alocada: 1 - 0 = 1 (custo máximo)
            #         # x_ivc = 1 se a UM i foi alocada ao veículo v para o cliente c e 0 se não foi alocada
            #         x_val.get((i["id"], v["id"], c["id"]), 0)
            #         for v in instancia["veiculos"]
            #         for c in instancia["clientes"]))
            #     for i in instancia["ums"]# Para cada UM
            # )

            # Detalhes das alocações por veículo
            # Para cada veículo, cria uma lista de IDs das UMs alocadas a ele
            for v in instancia["veiculos"]:
                cargas = [
                    i["id"] for i in instancia["ums"]
                    if any(x_val.get((i["id"], v["id"], c["id"]), 0) > 0.9  # ANY Verifica se a UM foi alocada ao veículo para qualquer cliente
                           for c in instancia["clientes"])
                ]

                if cargas:  # Se o veículo tem cargas alocadas a ele, calcula o valor total de:
                    peso_total = sum(i["peso"]
                                     for i in instancia["ums"] if i["id"] in cargas)
                    volume_total = sum(i["volume"]
                                       for i in instancia["ums"] if i["id"] in cargas)
                    frete_morto = z_val.get(v["id"], 0)

                    # Estrutura de Dados das Alocações
                    resultados['alocacoes'].append({
                        'veiculo_id': v["id"],
                        'veiculo_tipo': v["tipo"],
                        'destino': v["destino"],
                        'cargas': cargas,
                        'peso_total': peso_total,
                        'peso_minimo': v["carga_minima"],
                        'capacidade_peso': v["capacidade_peso"],
                        'volume_total': volume_total,
                        'capacidade_volume': v["capacidade_volume"],
                        'custo_veiculo': v["custo"],
                        'frete_morto': frete_morto,
                        'taxa_utilizacao_peso': (peso_total / v["capacidade_peso"]) * 100,
                        'taxa_utilizacao_volume': (volume_total / v["capacidade_volume"]) * 100
                    })

        if resultados and modelo.SolCount > 0:
            pasta_visualizacoes = os.path.join(
                os.path.dirname(__file__), 'Otimizacao', 'Visualizacoes')
            gerar_visualizacoes(resultados, instancia, pasta_visualizacoes)

        return resultados

    except Exception as e:
        print(f"❌ Erro ao processar instância {tipo_instancia}: {str(e)}")
        return None


def imprimir_resultados_detalhados(resultados):
    print(f"\n{'='*80}")
    print(
        f" 🟢 RESULTADOS PARA INSTÂNCIA: {resultados['tipo_instancia'].upper()}")
    print(f"{'='*80}")

    # Converter os códigos de status do Gurobi para texto
    status_map = {
        GRB.OPTIMAL: "Ótimo encontrado",
        GRB.TIME_LIMIT: "Tempo limite atingido",
        GRB.INFEASIBLE: "Problema inviável",
        GRB.INF_OR_UNBD: "Infinito ou ilimitado",
        GRB.UNBOUNDED: "Ilimitado"
    }

    print(
        f"\n🔷 Status: {status_map.get(resultados['status'], 'Desconhecido')}")
    print(f"⏳ Tempo de execução: {resultados['tempo_execucao']:.2f} segundos")

    if resultados['status'] == GRB.OPTIMAL:
        print(
            f"⏱️ Tempo para encontrar o ótimo: {resultados['tempo_para_otimo']:.2f} segundos")

    print(
        f"💰 Melhor solução encontrada: {resultados['melhor_solucao'] if resultados['melhor_solucao'] is not None else 'N/A'}")
    print(
        f"🔮 Solução relaxada: {resultados['solucao_relaxada'] if resultados['solucao_relaxada'] is not None else 'N/A'}")
    print(
        f"📊 GAP de otimização: {resultados['gap_otimizacao']:.2f}%" if resultados['gap_otimizacao'] is not None else "N/A")

    # versao anterior

    if resultados['status'] == GRB.OPTIMAL or resultados['status'] == GRB.TIME_LIMIT:
        # Função auxiliar para formatação segura
        def safe_format(value, fmt=".2f", prefix=""):
            return f"{prefix}{value:{fmt}}" if value is not None else "N/A"

        print(f"\n💵 CUSTOS:")
        print(
            f"  Total: {safe_format(resultados.get('custo_total'), '.2f', 'R$')}")
        print(
            f"  - Transporte: {safe_format(resultados.get('custo_transporte'), '.2f', 'R$')}")
        print(
            f"  - Frete morto: {safe_format(resultados.get('frete_morto_total'), '.2f', 'R$')}")
        print(
            f"  - Não alocação: {safe_format(resultados.get('custo_nao_alocacao'), '.2f', 'R$')}")

        print(f"\n🚚 VEÍCULOS:")
        print(f"  Ativos: {resultados.get('veiculos_ativos', 'N/A')}")
        print(f"  Inativos: {resultados.get('veiculos_inativos', 'N/A')}")

        # Detalhes por veículo
        for aloc in resultados.get('alocacoes', []):
            print(
                f"\n  Veículo {aloc.get('veiculo_id', 'N/A')} ({aloc.get('veiculo_tipo', 'N/A')} para {aloc.get('destino', 'N/A')}):")
            print(f"    Cargas: {aloc.get('cargas', 'N/A')}")
            print(f"    Peso: {safe_format(aloc.get('peso_total'), '.2f', '')}kg (min: {safe_format(aloc.get('peso_minimo'), '.2f', '')}kg, cap: {safe_format(aloc.get('capacidade_peso'), '.2f', '')}kg)")
            print(
                f"    Volume: {safe_format(aloc.get('volume_total'), '.2f', '')}m³ (cap: {safe_format(aloc.get('capacidade_volume'), '.2f', '')}m³)")
            print(
                f"    Utilização: {safe_format(aloc.get('taxa_utilizacao_peso'), '.1f', '')}% (peso), {safe_format(aloc.get('taxa_utilizacao_volume'), '.1f', '')}% (volume)")
            print(
                f"    Custo: {safe_format(aloc.get('custo_veiculo'), '.2f', 'R$')}")
            if aloc.get('frete_morto', 0) > 0:
                print(
                    f"    ℹ️ Frete morto: {safe_format(aloc.get('frete_morto'), '.2f', 'R$')}")

        # Cargas não alocadas
        print(f"\n📦 CARGAS NÃO ALOCADAS:")
        print(
            f"  Quantidade: {resultados.get('ums_nao_alocadas', 'N/A')} de {resultados.get('ums_alocadas', 0) + resultados.get('ums_nao_alocadas', 0)}")
        print(
            f"  Peso total: {safe_format(resultados.get('peso_nao_alocado'), '.2f', '')}kg")
        print(
            f"  Volume total: {safe_format(resultados.get('volume_nao_alocado'), '.2f', '')}m³")

        # Análise de decisões
        print(f"\n🔍 ANÁLISE DE DECISÕES:")
        if resultados.get('frete_morto_total', 0) > 0:
            print("  ℹ️ Há fretes mortos - veículos operando abaixo da capacidade mínima")
        else:
            print("  ✅ Nenhum frete morto - todos veículos atendem carga mínima")

        if resultados.get('ums_nao_alocadas', 0) > 0:
            print(
                f"  ℹ️ {resultados.get('ums_nao_alocadas', 0)} UMs não alocadas - verifique se é por restrições ou decisão ótima")
        else:
            print("  ✅ Todas UMs alocadas")

        if resultados.get('veiculos_inativos', 0) > 0:
            print(
                f"  ℹ️ {resultados.get('veiculos_inativos', 0)} veículos inativos - verifique se é esperado")

    print(f"\n{'='*80}")


def exportar_resultados_csv(resultados_lista, instancias_originais):

    # Exporta resultados para CSV

    #     resultados_lista: Lista de dicionários de resultados
    #     instancias_originais: Lista dos dados originais das instâncias
    #     caminho_saida: Pasta de destino (padrão no Google Drive)

    caminho_saida = os.path.join(os.path.dirname(
        __file__), 'Otimizacao', 'Resultados')

    # Verificação de segurança
    if not resultados_lista or not instancias_originais or len(resultados_lista) != len(instancias_originais):
        raise ValueError(
            "Listas de resultados e instâncias originais não correspondem")

    # Preparação do arquivo
    os.makedirs(caminho_saida, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    nome_arquivo = f"resultados_completos_{timestamp}.csv"
    caminho_completo = os.path.join(caminho_saida, nome_arquivo)

    with open(caminho_completo, mode='w', newline='', encoding='utf-8') as file:

        writer = csv.writer(file, delimiter=';')

        # Escreve cabeçalho principal
        writer.writerow(["RELATÓRIO DE OTIMIZAÇÃO"])
        writer.writerow(
            ["Gerado em:", datetime.now().strftime('%d/%m/%Y %H:%M:%S')])
        writer.writerow([])

        for resultados, instancia in zip(resultados_lista, instancias_originais):
            if not resultados or not instancia:
                continue

            # Verifica se a instância tem a estrutura esperada
            if 'ums' not in instancia or 'clientes' not in instancia:
                print(
                    f"⚠️ Estrutura inválida na instância {resultados.get('tipo_instancia', 'desconhecida')}")
                continue

            # Cabeçalho da instância
            writer.writerow(
                [f"INSTÂNCIA: {resultados.get('tipo_instancia', 'N/A')}"])
            writer.writerow([])

            writer.writerow([
                "Status", "Tempo Total (s)", "Tempo para Ótimo (s)",
                "Melhor Solução", "Solução Relaxada", "GAP (%)", "Custo Total",
                "Custo Transporte", "Frete Morto", "Custo Não Alocação",
                "Veículos Ativos", "Veículos Inativos", "UMs Alocadas", "UMs Não Alocadas",
                "Peso Não Alocado", "Volume Não Alocado"
            ])

            writer.writerow([
                "Ótimo" if resultados.get(
                    'status') == GRB.OPTIMAL else "Timeout",
                f"{resultados.get('tempo_execucao', 0):.2f}",
                f"{resultados.get('tempo_para_otimo', 0):.2f}" if resultados.get(
                    'tempo_para_otimo') is not None else "N/A",
                f"{resultados.get('melhor_solucao', 0):.2f}" if resultados.get(
                    'melhor_solucao') is not None else "N/A",
                f"{resultados.get('solucao_relaxada', 0):.2f}" if resultados.get(
                    'solucao_relaxada') is not None else "N/A",
                f"{resultados.get('gap_otimizacao', 0):.2f}" if resultados.get(
                    'gap_otimizacao') is not None else "N/A",
                f"{resultados.get('custo_total', 0):.2f}" if resultados.get(
                    'custo_total') is not None else "N/A",
                f"{resultados.get('custo_transporte', 0):.2f}",
                f"{resultados.get('frete_morto_total', 0):.2f}",
                f"{resultados.get('custo_nao_alocacao', 0):.2f}",
                resultados.get('veiculos_ativos', 0),
                resultados.get('veiculos_inativos', 0),
                resultados.get('ums_alocadas', 0),
                resultados.get('ums_nao_alocadas', 0),
                f"{resultados.get('peso_nao_alocado', 0):.2f}",
                f"{resultados.get('volume_nao_alocado', 0):.2f}"
            ])
            writer.writerow([])

            # Seção de alocações
            writer.writerow(["VEÍCULOS ATIVOS"])
            writer.writerow([
                "ID", "Tipo", "Destino", "Cargas", "Peso Total (kg)",
                "Capacidade (kg)", "Utilização (%)"
            ])

            for aloc in resultados.get('alocacoes', []):
                writer.writerow([
                    aloc.get('veiculo_id', ''),
                    aloc.get('veiculo_tipo', ''),
                    aloc.get('destino', ''),
                    ";".join(map(str, aloc.get('cargas', []))),
                    aloc.get('peso_total', ''),
                    aloc.get('capacidade_peso', ''),
                    f"{aloc.get('taxa_utilizacao_peso', 0):.1f}"
                ])
            writer.writerow([])

            # Seção de UMs não alocadas (com tratamento robusto)
            writer.writerow(["UNIDADES METÁLICAS NÃO ALOCADAS"])
            writer.writerow([
                "ID", "Tipo", "Peso (kg)", "Volume (m³)", "Cliente",
                "Destino", "Compatibilidade", "Motivo"
            ])

            # Obtém IDs alocados de forma segura
            alocados_ids = set()
            for aloc in resultados.get('alocacoes', []):
                alocados_ids.update(aloc.get('cargas', []))

            for um in instancia.get('ums', []):
                if um.get('id') not in alocados_ids:
                    # Encontra cliente associado
                    cliente = next(
                        (c for c in instancia.get('clientes', [])
                         if c.get('id') == um.get('cliente')),
                        {}
                    )

                    # Determina motivo
                    motivo = "Decisão ótima"
                    if not any(
                        v.get('tipo', '') in um.get(
                            'compatibilidade', '').split(',')
                        for v in instancia.get('veiculos', [])
                    ):
                        motivo = "Incompatibilidade"

                    writer.writerow([
                        um.get('id', ''),
                        um.get('tipo', ''),
                        um.get('peso', ''),
                        um.get('volume', ''),
                        cliente.get('nome', ''),
                        um.get('destino', ''),
                        um.get('compatibilidade', ''),
                        motivo
                    ])

            writer.writerow([])
            writer.writerow(["-"*50])
            writer.writerow([])

    print(f"\n✅ Relatório salvo em: {caminho_completo}")


def executar_todas_instancias_geradas():
    # Configurações
    PASTA_INSTANCIAS = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'Otimizacao')
    PASTA_RESULTADOS = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'Otimizacao', 'Resultados')
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)

    # Encontrar todos os arquivos CSV de instâncias
    arquivos_instancias = [f for f in os.listdir(PASTA_INSTANCIAS)
                           if f.endswith('.csv') and not f.startswith('00_')]

    if not arquivos_instancias:
        print("❌ Nenhuma instância encontrada na pasta!")
        return

    print(f"🔍 Encontradas {len(arquivos_instancias)} instâncias para executar")

    resultados_totais = []
    instancias_originais = []

    for arquivo in arquivos_instancias:
        try:
            nome_instancia = arquivo.replace('.csv', '')
            print(f"\n{'='*80}")
            print(f"🚀 PROCESSANDO INSTÂNCIA: {nome_instancia}")
            print(f"{'='*80}")

            # Carregar dados
            caminho_completo = os.path.join(PASTA_INSTANCIAS, arquivo)
            dados = carregar_dados(caminho_completo)
            instancia = {
                "veiculos": dados['veiculos'],
                "ums": dados['ums'],
                "clientes": dados['clientes'],
                "penalidade": dados['parametros']['Penalidade por não alocação']
            }
            instancias_originais.append(instancia)

            # Executar
            resultados = executar_instancia_com_timeout(
                nome_instancia, instancia)

            if resultados:
                resultados_totais.append(resultados)
                imprimir_resultados_detalhados(resultados)
            else:
                print(f"❌ Falha ao executar instância {nome_instancia}")

        except Exception as e:
            print(f"❌ Erro crítico ao processar {arquivo}: {str(e)}")
            continue

    # Exportar resultados consolidados
    if resultados_totais:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        nome_arquivo = f"resultados_consolidados_{timestamp}.csv"
        exportar_resultados_csv(resultados_totais, instancias_originais)
        print(
            f"\n✅ Todas instâncias processadas! Resultados em: {nome_arquivo}")
    else:
        print("\n⚠️ Nenhuma instância foi executada com sucesso!")


if __name__ == "__main__":

    executar_todas_instancias_geradas()
