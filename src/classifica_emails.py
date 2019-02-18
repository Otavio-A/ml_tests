import pandas as pd

classificacoes = pd.read_csv('../datasets/emails.csv')
textosPuros = classificacoes['email']
textosQuebrados = textosPuros.str.lower().str.split(' ')

palavras_desconhecidas = set()

for lista in textosQuebrados:
    palavras_desconhecidas.update(lista)

total_de_palavras = len(palavras_desconhecidas)

print(f"Total de palavras desconhecidas: {total_de_palavras}")


dicionario = {palavra:i for palavra, i in enumerate(palavras_desconhecidas)}
print(dicionario)
