import pickle

from Vocab import VocabClass
with open('./data/vocab.pkl', 'rb') as file:
    vocab: VocabClass = pickle.load(file)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
