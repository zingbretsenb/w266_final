from utils import nn_analogy_model_v3

model = nn_analogy_model_v3.nn_analogy_model_v3(embed_file="/data/w266_final/data/glove.6B.300d.txt")
model.buildModel(0.01, [900], use_dropout=True)
#model.buildModel(0.001, [16384,8192,4096,2048,1024,512], use_dropout=True)
model.trainModel(5000,5000,"/data/w266_final/data/questions-words.txt","/data/test_model")
results, scores = model.predict_from_file("/data/w266_final/data/predict-test.txt", "/data/test_model", return_scores=True)
print(results)
print(scores[0][model.word_to_id[results[0]]])
print(scores[0][model.word_to_id['queen']])
print(scores[0][model.word_to_id['woman']])
