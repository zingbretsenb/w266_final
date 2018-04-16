from utils import nn_analogy_model

model = nn_analogy_model.nn_analogy_model(embed_file="/home/fcampos/w266_final/data/glove.6B.300d.txt")
model.buildModel(0.001, [4096,2048,1024,512], use_dropout=True)
#model.trainModel(100, 500, "/home/fcampos/w266_final/data/google-test.txt", "/home/fcampos/test")
results = model.predict("/home/fcampos/w266_final/data/predict-test.txt", "/home/fcampos/test")
print(results)
