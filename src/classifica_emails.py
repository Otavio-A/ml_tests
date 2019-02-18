import numpy as np
import pandas as pd

from collections import Counter

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC

from utils import fit_and_predict

classificacoes = pd.read_csv('../datasets/emails.csv')
textos_puros = classificacoes['email']
textos_quebrados = textos_puros.str.lower().str.split(' ')

# todas as palavras desconhecidas
palavras_desconhecidas = set()

for lista in textos_quebrados:
    palavras_desconhecidas.update(lista)

total_de_palavras = len(palavras_desconhecidas)
print(f"Total de palavras desconhecidas: {total_de_palavras}")

# Associa um número para cada palavra
tuplas = list(zip(palavras_desconhecidas, range(total_de_palavras)))
dicionario = dict(tuplas)

def vetorizar_textos(texto, dicionario):
    """
        Transforma cada texto em um vetor com as ocorrências de cada palavra.
    """
    vetor = [0] * len(dicionario)
    for palavra in texto:
        if palavra in texto:
            posicao = dicionario[palavra]
            vetor[posicao] += 1

    return vetor

# vetoriza os textos
vetores_de_texto = [vetorizar_textos(texto, dicionario) for texto in textos_quebrados]

# Machine Learning

marcacoes = classificacoes['classificacao']

X = vetores_de_texto
Y = marcacoes

porcentagem_de_treino = 0.8

tamanho_de_treino = int(porcentagem_de_treino * len(Y))

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

validacao_dados = X[tamanho_de_treino:]
validacao_marcacoes = Y[tamanho_de_treino:]



