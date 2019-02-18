import numpy as np
from sklearn.model_selection import cross_val_score

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
    """
        Utiliza o k-fold para fazer o treino e a previsão do modelo

        Args:
            nome: Nome do modelo
            modelo: Modelo do algorimto
            treino_dados: Dados de treino
            treino_marcacoes: Marcações de treino
        Return:
            taxa_de_acerto: A taxa de acerno do modelo do algoritmo
    """
    k = 10
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv=k)
    taxa_de_acerto = np.mean(scores)

    msg = f"Taxa de acerto do {nome} : {taxa_de_acerto}"
    print(msg)
    return taxa_de_acerto