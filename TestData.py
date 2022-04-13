import torch
from Synonyms import get_synonyms_list
from transformers import BertTokenizer, BertModel
import numpy as np
import bert_encoder
import torch
import argparse
from config2 import load_hyperparam

soft_positions = [] # lijst met per zin een lijst van softpositions van de ontokenized zinnen
new_soft_positions = []
Tokens_zinnen = []
segments = []
visible_matrices  = []
is_original = []
new_originals = []
model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True )
model.eval()
def get_sentence_with_synonyms(synonyms, sent):
    sentence_list = sent.split(" ")
    copy_sentence = sentence_list.copy()

    counter = 0
    cnt = 0
    original = []
    soft_position = []

    for word in copy_sentence:
        counter += 1
        original.append(1)

        if synonyms ==None:
            pass
        else:
            list_synonyms = synonyms.get_lex_representations_without_himself(word)
            sentence_list[counter:counter] = list_synonyms
            counter += len(list_synonyms)


        soft_position.append(cnt)


        if synonyms ==None:
            pass
        else:
            for i in range(0, len(" ".join(list_synonyms).split(" "))):
        
                if " ".join(list_synonyms).split(" ")[0] == '':
                    pass
                else:
                    soft_position.append(cnt)
                    original.append(0)


        cnt += 1
    is_original.append(original)
    soft_positions.append(soft_position)



    return sentence_list


def divide_words_in_sentence(zin, tokenizer, sent): # hier voor elk woord tokenizen en aantal len van de nieuwe lijst aan zelfde soft positions toevoegen
    new_soft_position = []
    new_original = []
    if zin!=None:

        for i in range(0,len(sent.split(" "))):

            tok = tokenizer.tokenize(sent.split(" ")[i])

            pos = soft_positions[zin][i]
            ori = is_original[zin][i]
            for j in range(0,len(tok)):
                new_original.append(ori)
                new_soft_position.append(pos)
        new_originals.append(new_original)
        new_soft_positions.append(new_soft_position)
        list_with_dividing = tokenizer.tokenize(sent)
        count1 = 0
        count = 0
        for i in range(0,len(new_soft_positions[zin])): # makes softpositions unique for tokens for original word and not unique for tokens of synonyms



            if new_original[i]== 1:

                count = 0
                new_soft_positions[zin][i] = count1
                count1 += 1
            elif new_original[i] == 0:

                new_soft_positions[zin][i] = i - count
                count+=1
        sentence = ' '.join(list_with_dividing)
        Tokens_zinnen.append(sentence.split(" "))
               # hierboven worden de softpositions toegevoegd aan een nieuwe lijst, deze wordt toegevoegd aan de grote lijst voor elke zin
    else:
        list_with_dividing = tokenizer.tokenize(sent)
        sentence = ' '.join(list_with_dividing)
         # voeg alle tokens in lijst vorm aan de grote lijst
    sentence2 = ''
    if '$ t $' in sentence:
        sentence2 = sentence.replace('$ t $', '$T$')
    else:
        sentence2 = sentence
    return sentence2

def makeSegments():
    for sentence in Tokens_zinnen: # itereert over alle zinnen, dan over alle tokens en zet segment id's neer en stuurt lijst naar een grotere lijst
        seg = []
        s_count = 0

        for token in sentence:

            if s_count  == 0:
                seg.append(0) #can change it to zero
            elif s_count == 1:
                seg.append(1)
            elif s_count %2 == 0:
                seg.append(0)
            elif s_count %2 != 0:
                seg.append(1)
               

            if token == "-" or token=='–':

                s_count += 1

        segments.append(seg)

def makeVisibelMatrices():
    for positions,original in zip(new_soft_positions,new_originals):
        visible_matrices.append(get_visible_matrix(positions,original))

def get_visible_matrix(positions,original):
        '''
        gets the visible matrix of a sentence tree
        '''
        visible_matrix = np.zeros((1,len(positions), len(positions)))
        for i in range(0,len(positions)):
            for j in range(0, len(positions)):
                visible_matrix[0,i, j] = float(determine_visibility(original,positions,i, j))
        return visible_matrix

def determine_visibility(original,positions, row_number, column_number):
    '''
    method that determines if a row number and column number can see each other in the sentence tree
    '''
    result = 0
    row_is_first_occurrence = is_first_occurrence(original,positions, row_number)
    column_is_first_occurrence = is_first_occurrence(original,positions, column_number)
    if not row_is_first_occurrence or not column_is_first_occurrence:
        result = -10000.0
    if same_soft_position(positions,row_number, column_number):
        result = 0

    return result

def same_soft_position(positions, position1, position2):
    '''
    method that determines if two soft positions are the same or not
    '''
    if positions[position1] == positions[position2]:
        return True

    return False

def is_first_occurrence(original, positions,number):
    '''
    method that determines if the soft position is the first occurrence in the sentence tree
    '''
    soft_position_before = -1
    soft_position_number = positions[number]
    if number != 0:
        soft_position_before = positions[number - 1]
    if soft_position_before == soft_position_number and original[number] == 0:
        return False

    return True


class TestData:

    list_of_synonyms = get_synonyms_list()
    print(list_of_synonyms.my_dict)
    print(list_of_synonyms.list_of_list_of_synonyms)

# tijdens het synoniemen toevoegen en tokenizen ook soft positions maken en in een lijst opslaan of gelijk elke zin een tensor geven met de juiste embeddings.
# zelfde geldt voor de segments en tokens, dus hieronder alle embeddings geven, de visible matrix uitrekenen en alles in tensors zetten.

    count = 0
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with open('data/externalData/raw_data2016.txt', 'r') as raw_data:
        line_list = raw_data.readlines()

        with open('testdata2016.txt', 'w') as test_data:
            # append the raw training data
            zin = 0
            for i in range(0, 5640):
                sentence = line_list[i]
                if count % 3 == 0:
                    sentence = "[CLS] " + line_list[i].replace('$T$', line_list[i+1].replace('\n', '')) + " [SEP]"
                    sentence_without_synonyms = get_sentence_with_synonyms(None, sentence)
                    sentence2 = ' '.join(sentence_without_synonyms)
                    sentence_with_dividing = divide_words_in_sentence(zin,tokenizer, sentence2)
                    test_data.write(sentence_with_dividing + '\n')
                    zin += 1
                else:
                    sentence_with_dividing = divide_words_in_sentence(None, tokenizer, sentence)
                    test_data.write(sentence_with_dividing + '\n')
                count+=1
            for i in range(5640, len(line_list)):

                sentence = line_list[i]

                if count % 3 == 0:
                    sentence = "[CLS] " + line_list[i].replace('$T$', line_list[i+1].replace('\n', '')) + " [SEP]"

                    synonym = get_synonyms_list()
                    sentence_with_synonyms = get_sentence_with_synonyms(synonym, sentence)

                    sentence2 = ' '.join(sentence_with_synonyms)
                    sentence_with_synonyms_and_dividing = divide_words_in_sentence(zin,tokenizer, sentence2)
                    test_data.write(sentence_with_synonyms_and_dividing + '\n')
                    zin += 1
                else:
                    sentence_with_dividing = divide_words_in_sentence(None, tokenizer, sentence)
                    test_data.write(sentence_with_dividing + '\n')
                count += 1

embeddings = []
visibleMatrices = []
def makeEmbeddings():

    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    token_id_embeddings = model.embeddings.word_embeddings.weight
    seg_id_embeddings = model.embeddings.token_type_embeddings.weight
    pos_id_embeddings = model.embeddings.position_embeddings.weight
    
    for i in range(0,len(new_soft_positions)):
        tensor = torch.zeros((1,len(new_soft_positions[i]),768))

        for j in range(0, len(new_soft_positions[i])):

            tensor[0][j] = token_id_embeddings[tokenizer.convert_tokens_to_ids(Tokens_zinnen[i][j])] \
                           + seg_id_embeddings[segments[i][j]] \
                           + pos_id_embeddings[new_soft_positions[i][j]]

        embeddings.append(tensor)

        visibleMatrices.append(torch.tensor(visible_matrices[i]))



makeSegments() # na alle tokens hebben toegevoegd maak segments.
makeVisibelMatrices()
print('here')
makeEmbeddings()





        
        

# code hieronder telt alle unieke tokens die file ook naar k bert sturen, eerst alle unieke tokens in lijsten zetten en per zin een lijst weer.

all_whole_sentence = []
all_counted_tokens = []
count = -1
count2 = 1
with open('testdata2016.txt', 'r') as test_data1:
    line_list = test_data1.readlines()
    for i in range(0, len(line_list)):
        sentence_list = line_list[i].split(" ")
        sentence_list_without_next_line = []
        for j in range(0, len(sentence_list)):
            word = sentence_list[j]
            if '\n' in word:
                word2 = word.replace('\n', '')
            else:
                word2 = word
            sentence_list_without_next_line.append(word2)
        all_whole_sentence.append(sentence_list_without_next_line)
    with open('test_data2016.txt', 'w') as test_data_unique:
        dic_words = {}
        for i in range(0, 5640):
            count += 1
            count2 += 1
            if count % 3 == 0 : # removed the or statement of count2% 3==0
                for j in range(0, len(all_whole_sentence[i])):
                    if all_whole_sentence[i][j] =='$T$'or  all_whole_sentence[i][j] =='[SEP]' or all_whole_sentence[i][j] =="[CLS]":
                        pass
                    elif not all_whole_sentence[i][j] in dic_words:
                        dic_words[all_whole_sentence[i][j]] = 0
                    else:
                        past_value = dic_words[all_whole_sentence[i][j]]
                        dic_words[all_whole_sentence[i][j]] = past_value + 1

                    if not all_whole_sentence[i][j] == '$T$' and all_whole_sentence[i][j] !='[SEP]' and all_whole_sentence[i][j] !="[CLS]":
                        all_whole_sentence[i][j] = all_whole_sentence[i][j] + '_' + str(dic_words[all_whole_sentence[i][j]])
                all_counted_tokens.append(all_whole_sentence[i])
                sent = ' '.join(all_whole_sentence[i])
                test_data_unique.write(sent + '\n')
            else:
                test_data_unique.write(line_list[i])
        for i in range(5640,len(all_whole_sentence)):
            count += 1
            count2 += 1
            if count % 3 == 0:
                for j in range(0, len(all_whole_sentence[i])):
                    if all_whole_sentence[i][j] == '$T$'or all_whole_sentence[i][j] =='[SEP]' or all_whole_sentence[i][j] =="[CLS]":
                        pass
                    elif not all_whole_sentence[i][j] in dic_words:
                        dic_words[all_whole_sentence[i][j]] = 0
                    else:
                        past_value = dic_words[all_whole_sentence[i][j]]
                        dic_words[all_whole_sentence[i][j]] = past_value + 1

                    if not all_whole_sentence[i][j] == '$T$'and all_whole_sentence[i][j] !='[SEP]' and all_whole_sentence[i][j] !="[CLS]":
                        all_whole_sentence[i][j] = all_whole_sentence[i][j] + '_' + str(dic_words[all_whole_sentence[i][j]])
                all_counted_tokens.append(all_whole_sentence[i])
                sent = ' '.join(all_whole_sentence[i])
                test_data_unique.write(sent + '\n')
            else:
                test_data_unique.write(line_list[i])


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_path", default="./google_config.json", type=str,help="Path of the config file.")
args = parser.parse_args()


args = load_hyperparam(args) # Load the hyperparameters from the config file.
encoder = bert_encoder.BertEncoder(args, model) # maak object bert encoder aan en roep forward functie op met emb, seg en visible matrix. seg kan volgens mij ook none dat is voor aspect based classification.

Tokens = all_counted_tokens # Lijst van lijsten waarbij de lijst alle tokens voor 1 zin bevat. eerste element is dus voor eerste zin.
VM = visibleMatrices # een lijst met voor elke zin een visible matrix in tensor-vorm van 1*token_numb*token_numb.
Embeddings = embeddings # een lijst met voor elke zin een tensor van 1*token_numb*768 met de initiële embeddings voor elke token op volgorde van hoe ze in de zin voorkomen.

hidden_states = [] # verzamel alle hidden states als het goed is op volgorde van Tokens lijst.
token_hidden_states = [] # lijst met voor elke token de token en daarachter de hidden states.
#with open('testEmbeddings.txt','w') as outf:
    #outf.truncate()

for i in range(2000,len(Tokens)):
    hidden = encoder.forward(Embeddings[i], None, VM[i]) # reken hidden states uit voor alle tokens in 1 zin.
    hidden_states.append(hidden) # voeg aan een lijst toe voor later gebruik
    print(i)


counter = 0
for j in range(2000,len(Tokens)): # itereer over alle zinnen
    print( j)
    token_count = 0  # tel hoeveel tokens je heb gehad in de zin van Tokens lijst.
    for token in Tokens[j]: # itereer over alle tokens per zin

        if token == "[CLS]" or token == "[SEP]":

            token_count += 1
        else:
            list_of_embeddings = hidden_states[counter][0][token_count].tolist() # pak de embedding voor de token en zet naar een lijst
            token_count += 1 # tel tokens
            string_list_of_embeddings = [str(i) for i in list_of_embeddings] # Maak van alle getallen string object
            string_list_of_embeddings.insert(0, token) # zet token op eerste plek van lijst
            token_hidden_states.append(string_list_of_embeddings) # zet de hele lijst in een andere lijst
    counter += 1
with open('testEmbeddings4.txt','w') as outf:

    for c in token_hidden_states:

        print(" ".join(c), file= outf)        # print alle embeddings naar een txt file
    outf.close()

''''
with open('test_data2016.txt', 'r') as td:
    print('printing test data2016')
    line_list = td.readlines()
    with open('test_data2016_final.txt', 'w') as final:
        for i in range (0, len(line_list)):
             final.write(line_list[i])

with open('test_data2016_final.txt', 'r') as fin:

    line_list = fin.readlines()
    target_list = []
    for i in range(1, len(line_list), 3):
         target_list.append(line_list[i])
    with open('test_data2016_sentences.txt', 'w') as test_d:
        for i in range(0, len(line_list), 3):
            sentence2 = ''
            sentence2 = line_list[i].replace('$T$', line_list[i+1].replace('\n', ''))
            test_d.write(sentence2)
'''''
