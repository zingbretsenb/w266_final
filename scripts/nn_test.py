from utils import nn_analogy_model

model = nn_analogy_model.nn_analogy_model(embed_file="/home/fcampos/w266_final/data/glove.6B.300d.txt")
model.buildModel(0.0001, [16384,8192,4096,2048,1024,512], use_dropout=True)
model.trainModel(200,1000,"/home/fcampos/w266_final/data/questions-words.txt","/home/fcampos/test")
results, scores = model.predict_from_file("/home/fcampos/w266_final/data/predict-test.txt", "/home/fcampos/test", return_scores=True)
print(results)
print(scores[0][model.word_to_id[results[0]]])
print(scores[0][model.word_to_id['queen']])
print(scores[0][model.word_to_id['woman']])
