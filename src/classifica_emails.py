import numpy as np
import pandas as pd

from collections import Counter

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC
import nltk

from utils import fit_and_predict_kfold, teste_real, vetorizar_textos

# nltk.download('tokenize')

classificacoes = pd.read_csv('../datasets/emails.csv')
textos_puros = classificacoes['email']
frases = textos_puros.str.lower()
textos_quebrados = [nltk.tokenize.word_tokenize(frase) for frase in frases]

#nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')
#nltk.download('rslps')
stemmer = nltk.stem.RSLPStemmer()   # obj para tirar a raiz das palavras

# todas as raizes das palavras desconhecidas (sem stopwords)
palavras_desconhecidas = set()
for lista in textos_quebrados:
    palavras_validas = [stemmer.stem(palavra) for palavra in lista if not palavra in stopwords and len(palavra) > 2]
    palavras_desconhecidas.update(palavras_validas)

total_de_palavras = len(palavras_desconhecidas)
print(f"Total de palavras desconhecidas: {total_de_palavras}")

# Associa um número para cada palavra
tuplas = list(zip(palavras_desconhecidas, range(total_de_palavras)))
dicionario = dict(tuplas)

# vetoriza os textos
vetores_de_texto = [vetorizar_textos(texto, dicionario, stemmer) for texto in textos_quebrados]


# Machine Learning
marcacoes = classificacoes['classificacao']

X = vetores_de_texto
Y = marcacoes

porcentagem_de_treino = 0.8

tamanho_de_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_validacao = len(Y) - tamanho_de_treino

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

validacao_dados = X[tamanho_de_treino:]
validacao_marcacoes = Y[tamanho_de_treino:]


# Testas os modelos
resultados = dict() # guarda os resultados dos modelos

# Teste com OneVsRestClassifier
modelo_one_vs_rest = OneVsRestClassifier(LinearSVC(random_state = 0, max_iter=10000))
resultado_one_vs_rest = fit_and_predict_kfold("OneVsRest", modelo_one_vs_rest, treino_dados, treino_marcacoes)
resultados[resultado_one_vs_rest] = {'modelo': modelo_one_vs_rest, 'nome': "OneVsRest"}

# Teste com OneVsOneClassifier
modelo_one_vs_one = OneVsRestClassifier(LinearSVC(random_state = 0, max_iter=10000))
resultado_one_vs_one = fit_and_predict_kfold("OneVsOne", modelo_one_vs_one, treino_dados, treino_marcacoes)
resultados[resultado_one_vs_one] = {'modelo': modelo_one_vs_one, 'nome': "OneVsOne"}

# Teste com MultinomialNB
modelo_multinomialnb = MultinomialNB()
resultado_multinomialnb = fit_and_predict_kfold("MultinomialNB", modelo_multinomialnb, treino_dados, treino_marcacoes)
resultados[resultado_multinomialnb] = {'modelo': modelo_multinomialnb, 'nome': "MultinomialNB"}

# Teste com AdaBoostClassifier
modelo_ada_boost = AdaBoostClassifier(random_state=0)
resultado_ada_boost = fit_and_predict_kfold("AdaBoostClassifier", modelo_ada_boost, treino_dados, treino_marcacoes)
resultados[resultado_ada_boost] = {'modelo': modelo_ada_boost, 'nome': "AdaBoostClassifier"}

# Seleciona o modelo que teve maior resultado nos testes
maior_resultado = max(resultados)
vencedor = resultados[maior_resultado]

print(f"O vecendor foi: {vencedor['nome']}")
vencedor['modelo'].fit(treino_dados, treino_marcacoes)

# Testa o modelo vencedor com dandos que ele desconhece
teste_real(vencedor['nome'], vencedor['modelo'], validacao_dados, validacao_marcacoes)

# Taxa de acerto do algoritmo que realiza chutes com base
# na maior quantidades de labels na lista de validação
acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100*(acerto_base/len(validacao_marcacoes))
print(f"A taxa de acerto base foi: {taxa_de_acerto_base}")

total_de_elementos = len(validacao_dados)
