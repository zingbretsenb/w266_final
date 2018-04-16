from utils import nn_analogy_model

model = nn_analogy_model.nn_analogy_model(embed_file="/home/fcampos/w266_final/data/glove.6B.300d.txt")
model.buildModel(0.1, [4096,2048,300], use_dropout=True)
model.trainModel(40, 1000, "/home/fcampos/w266_final/data/questions-words.txt", "/home/fcampos/test")
logits = model.predict("/home/fcampos/w266_final/data/google-test.txt", "/home/fcampos/test")
