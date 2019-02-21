import pandas as pd
import numpy as np
from collections import Counter

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score

from utils import fit_and_predict_kfold, teste_real

df = pd.read_csv('../datasets/situacao_do_cliente.csv')
X_df = df[['recencia','frequencia', 'semanas_de_inscricao']]
Y_df = df['situacao']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values 

porcentagem_de_treino = 0.8

tamanho_de_treino = int(porcentagem_de_treino * len(Y))

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
modelo_ada_boost = AdaBoostClassifier()
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

print(f"Total de elementos testados: {total_de_elementos}")
