import numpy as np
from sklearn.model_selection import cross_val_score


def fit_and_predict(nome, modelo, trenio_dados, treino_marcacoes,
                    teste_dados, teste_marcacoes):
    
    """
    Faz o treino e o teste do algorimto
    """

    # Treina o modelo
    modelo.fit(trenio_dados, treino_marcacoes)

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

def fit_and_predict_kfold(nome, modelo, treino_dados, treino_marcacoes):
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


def teste_real(nome, modelo, validacao_dados, validacao_marcacoes):
    """
        Valida o modelo escolhido como vencedor, com dados que nunca viu

        Args:
            nome: Nome do modelo vencedor
            modelo: Modelo do algoritmo que foi escolhido como vencedor
            validacao_dados: Dados inéditos para o modelo vencedor irá prever
            validacao_marcacao: Marcações usadas para validar o modelo vencedor
    """

    resultado = modelo.predict(validacao_dados)
    acertos = (resultado == validacao_marcacoes)

    total_de_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)
    
    taxa_de_acertos = 100*(total_de_acertos/total_de_elementos)
    taxa_de_acertos = format(taxa_de_acertos, '.2f')

    msg = f"Algoritmo vencedor: {nome}. Taxa de acerto do algorimto: {taxa_de_acertos}%"
    print(msg)

def vetorizar_textos(texto, dicionario, stemmer):
    """
        Transforma cada texto em um vetor com as ocorrências de cada palavra.
    """
    vetor = [0] * len(dicionario)
    for palavra in texto:
        if len(palavra) > 0:
            raiz_palavra = stemmer.stem(palavra)
            if raiz_palavra in dicionario:
                posicao = dicionario[raiz_palavra]
                vetor[posicao] += 1

    return vetor