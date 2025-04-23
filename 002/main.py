import pandas as pd
import requests
import json
from enelvo import normaliser

df = pd.read_csv("dados_test.csv")["Conteúdo"]

norm = normaliser.Normaliser()
norm_sentences = []
for sentence in df:
    norm_sentences.append(norm.normalise(sentence))

url = "https://api-nice.ibict.br/ibictLLM"
headers = {
    'Content-Type': 'application/json',
    'apikey':'',}

payload = {'model':'base1.1:8b',
          'stream': False,
          'keep_alive':'60m',
          "options":{
            "seed":105,
            "temperature":0.1,
            "num_ctx": 8192
            },

          "messages": [
              {
                "role": "system",
                "content":"A partir de agora você deve agir como um analista de sentimentos. Você vai me devolver o sentimento relativo aos dados de textos que serão mandados para você de acordo com uma criteriosa análise de linguagem. Separe o conteudo do texto e os sentimentos com um separador"
              },
                    {
                "role": "assistant",
                "content": "Eu sou um analista de sentimentos que pega um conjunto de sentimentos e separa da seguinte forma: 'conteudo','sentimentos'."
              },
              {
                "role": "user",
                "content": f"{norm_sentences}"
              },
            ]
          }

response = requests.post(url, json=payload, headers=headers)
conteudo = response.content.decode("utf-8")
data = json.loads(conteudo)
mensagem = data["message"]["content"]
print(mensagem)
