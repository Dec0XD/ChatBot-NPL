import nltk
import tflearn
import numpy as np
import json

# ...imports e preprocessamento de dados...

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

# ...inicialização do Flask...

model = tflearn.DNN(net)
model.load("model.tflearn")

# ...rotas do Flask...


# Carregue o arquivo JSON de intenções
with open("intents.json") as file:
    data = json.load(file)

# Defina uma função para processar a entrada do usuário e retornar a resposta do bot
def get_bot_response(user_text):
    # Tokenize a entrada do usuário
    tokens = nltk.word_tokenize(user_text.lower())

    # Use o modelo para prever a intenção da entrada do usuário
    results = model.predict([tokens])[0]

    # Escolha a intenção com a maior probabilidade
    intent_idx = np.argmax(results)
    intent = data["intents"][intent_idx]

    # Escolha uma resposta aleatória para a intenção escolhida
    response = np.random.choice(intent["responses"])

    return response

# Teste a função
print(get_bot_response("Olá, como você está?"))
