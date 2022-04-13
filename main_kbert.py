import argparse
from config2 import load_hyperparam
import bert_encoder
import torch
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_path", default="./google_config.json", type=str,help="Path of the config file.")
args = parser.parse_args()


args = load_hyperparam(args) # Load the hyperparameters from the config file.
encoder = bert_encoder.BertEncoder(args) # maak object bert encoder aan en roep forward functie op met emb, seg en visible matrix. seg kan volgens mij ook none dat is voor aspect based classification.

Tokens = [] # Lijst van lijsten waarbij de lijst alle tokens voor 1 zin bevat. eerste element is dus voor eerste zin.
VM = [] # een lijst met voor elke zin een visible matrix in tensor-vorm van 1*token_numb*token_numb.
Embeddings = [] # een lijst met voor elke zin een tensor van 1*token_numb*768 met de initiële embeddings voor elke token op volgorde van hoe ze in de zin voorkomen.

hidden_states = [] # verzamel alle hidden states als het goed is op volgorde van Tokens lijst.
token_hidden_states = [] # lijst met voor elke token de token en daarachter de hidden states.

for i in range(0,len(VM)):

    hidden = encoder.forward(Embeddings[i], None, VM[i]) # reken hidden states uit voor alle tokens in 1 zin.
    hidden_states.append(hidden) # voeg aan een lijst toe voor later gebruik


for j in range(0,len(Tokens)): # itereer over alle zinnen

    token_count = 0  # tel hoeveel tokens je heb gehad in de zin van Tokens lijst.
    for token in Tokens[j]: # itereer over alle tokens per zin

        list_of_embeddings = hidden_states[j][0][token_count].tolist() # pak de embedding voor de token en zet naar een lijst
        token_count += 1 # tel tokens
        string_list_of_embeddings = [str(i) for i in list_of_embeddings] # Maak van alle getallen string object
        string_list_of_embeddings.insert(0, token) # zet token op eerste plek van lijst
        token_hidden_states.append(string_list_of_embeddings) # zet de hele lijst in een andere lijst

with open('testEmbeddings.txt','w') as outf:

    outf.truncate()
    for c in token_hidden_states:

        print(" ".join(c), file= outf)        # print alle embeddings naar een txt file







    

print(outf)
