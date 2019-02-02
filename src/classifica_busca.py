import pandas as pd
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier


def fit_and_predict(nome, modelo, trenio_dados, treino_marcacoes,
                    teste_dados, teste_marcacoes):
    
    """
    Faz o treino e o teste do algorimto
    """

    # Treina o modelo
    modelo.fit(treino_dados, treino_marcacoes)

    # Resultado da predição
    resultado = modelo.predict(teste_dados)
    # Pega os dados que o modelo acertou
    acertos = (resultado == teste_marcacoes)
    # Total de acertos
    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_dados)

    # Eficacia do modelo
    taxa_de_acertos = 100.0*(total_de_acertos/total_de_elementos)
    taxa_de_acertos = format(taxa_de_acertos, '.2f')

    # Resultado
    msg = f'Taxa de acerto do algoritmo {nome} : {taxa_de_acertos}%'
    print(msg)

    return taxa_de_acertos

def teste_real(nome, modelo, validacao_dados, validacao_marcacoes):
    
    """
    Testa o algoritmo que teve o melhor resultado nos testes.
    """

    resultado = modelo.predict(validacao_dados)
    acertos = (resultado == validacao_marcacoes)

    total_de_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)

    taxa_de_acertos = 100*(total_de_acertos/total_de_elementos)
    taxa_de_acertos = format(taxa_de_acertos, '.2f')

    msg = f'Algoritmo vencedor: {nome}. Taxa de acerto do algoritmo: {taxa_de_acertos}%'
    print(msg)

# teste inicial: home, busca, logado => comprou
# home, busca
# home, logado
# busca, logado
# busca: 85,71% (8 testes)

# Lê o arquivo.
df = pd.read_csv('../datasets/buscas.csv')

# Data Frame
## A coluna 'busca' tem mais relevância na classificação.
X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

# Transforma as varíaveis categóricas em binário.
Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

# Pega os valores inteiros.
X = Xdummies_df.values
Y = Ydummies_df.values

# Quantidade de dados usada para treino e teste.
porcentagem_de_treino = 0.8
porcentagem_de_teste = 0.1
tamanho_de_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_teste = int(porcentagem_de_teste * len(Y))
tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste

# Dados que serão usadas para treinar.
treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

# Dados para testar o modelo.
fim_de_teste = tamanho_de_treino + tamanho_de_teste
teste_dados = X[tamanho_de_treino:fim_de_teste]
teste_marcacoes = Y[tamanho_de_treino:fim_de_teste]

# Dados para validar o modelo que
# se saiu melhor nos testes.
validacao_dados = X[fim_de_teste:]
validacao_marcacoes = Y[fim_de_teste:]

# Teste com o MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados,
treino_marcacoes, teste_dados, teste_marcacoes)

# Teste com o AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados,
treino_marcacoes, teste_dados, teste_marcacoes)

if resultadoAdaBoost > resultadoMultinomial:
    vencedor, nome = modeloAdaBoost, "AdaBoostClassifier"
else:
    vencedor, nome = modeloMultinomial, "MultinomialNB"

teste_real(nome, vencedor, validacao_dados, validacao_marcacoes)

# eficacia do algoritmo que chuta só 1 ou 0 (algoritmo burro)
## Usa o valor que mais se repete no conjunto de testes para chutar
acerto_base = max(Counter(teste_marcacoes).values())
# Calcula a taxa de acerto
taxa_de_acerto_base = 100.0 * (acerto_base/tamanho_de_teste)
taxa_de_acerto_base = format(taxa_de_acerto_base, '.2f')

total_de_elementos = len(validacao_dados)

# Resultado
print(f"Taxa de acerto base: {taxa_de_acerto_base}%")
print(f'Total de elementos testados: {total_de_elementos}')

