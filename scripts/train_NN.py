from utils import nn_analogy_model

model = nn_analogy_model.nn_analogy_model(embed_file="/home/fcampos/w266_final/data/glove.6B.300d.txt")
model.buildModel(0.001, [4096,2048,1024,512], use_dropout=True)
model.trainModel(100, 10000, "/home/fcampos/w266_final/data/BATS_3.0/full_training_file.txt", "/home/fcampos/w266_model")
