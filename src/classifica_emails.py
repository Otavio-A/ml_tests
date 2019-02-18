import numpy as np
import pandas as pd

from collections import Counter

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
