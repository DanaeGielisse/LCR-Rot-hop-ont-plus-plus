from config import *

with open('data/temporaryData/' + 'BERT768embedding' + str(FLAGS.year) + '.txt', 'r') as bertem:
    line_list = bertem.readlines()
with open('data/temporaryData/' + 'test_embeddings' + str(FLAGS.year) + '.txt', 'r') as testem:
    line_list2 = testem.readlines()
    with open('data/programGeneratedData/' + 'embeddings' + str(FLAGS.year) + '.txt', 'w') as em:
        for line in line_list:
            if line.__contains__("serves_3"):
                break
            else:
                em.write(line)
        for line in line_list2:
            em.write(line)

