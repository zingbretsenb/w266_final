from utils import nn_analogy_model, data

sat_data = data.FileFinder().get_sat_data()
model = nn_analogy_model.nn_analogy_model(embed_file="/home/fcampos/w266_final/data/glove.6B.300d.txt")
model.buildModel(0.001, [4096,2048,1024,512], use_dropout=True)
score = 0
questions = 0
oov_questions = 0
index_to_letter = {0: 'a', 1:'b', 2:'c', 3:'d', 4:'e'}
print("Running questions...",flush=True)
for question in sat_data:
    questions += 1
    alternatives = []
    d = []
    for answer in question['answers']:
        alternatives.append([question['question'][0],question['question'][1],answer[0][0]])
        d.append(answer[0][1])
    results, oov, scores = model.predict(alternatives, "/home/fcampos/w266_model", return_scores = True)
    choice_index = []
    for i in range(len(question['answers'])):
        if i not in oov:
            if results[i] == d[i]:
                choice_index.append(i)
    if len(choice_index) == 0 and len(oov) == 5:
        print("Question %d: Out-of-vocabulary words in question or all alternatives" % questions)
    elif len(choice_index) == 0 and len(oov) < 5:
        print("Question %d: Couldn't find viable alternative. Correct: %s. Alternatives:" % (questions,question['correct'][0][1]))
        print(results)
    elif len(choice_index) == 1:
        if index_to_letter[i] == question['correct_letter']:
            print("Question %d: Correct answer found" % questions)
            score += 1
        else: 
            print("Question %d: Incorrect answer found" % questions)
    else:
        correct = False
        for choice in choice_index:
            if index_to_letter[choice] == question['correct_letter']:
                correct = True
        if correct:
            print("Question %d: Multiple answers found, correct among them" % questions)
            score += 1
        else:
            print("Question %d: Multiple answers found, none of them correct" % questions)
print("\n\n-----------------------------------")
print("Summary Statistics")
print("\n")
print("Number of questions read: %d" % questions)
print("Number of correct answers: %d" % score)
print("Final Score: %.3f%%" % ((score/questions)*100))
print("-----------------------------------")
