import csv
import re
from math import sqrt
import pickle
import torch
from model.Recognizer import AL_net
import Vocab
from Vocab import VocabClass
from utils.rna2sorf_csv2csv import get_index_section, sorf_to_protein

with open('./data/vocab.pkl', 'rb') as file:
    vocab: VocabClass = pickle.load(file)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Recognizer:
    def __init__(self, embedding_dim=128, batch_size=64, vocab_size=68, token_len=102, hidden_size=256, epoch=5,
                 dropout=0):
        self.al_net = AL_net(embedding_dim, batch_size, vocab_size, token_len, hidden_size, dropout)
        self.al_net.to(device)
        self.vocab = vocab

    def load_train_model(self, load_path, filename):
        if len(load_path) == 0:
            print("[*] Checkpoint not found! Starting from scratch.")
            return 0  # file is not there
        al_weights = torch.load(load_path + "{}".format(filename), map_location=device)
        self.al_net.load_state_dict(al_weights)

    def predict(self, valid_seq):
        sigmoid = torch.nn.Sigmoid()
        words = [valid_seq[i * 3:(i + 1) * 3] for i in range(int(len(valid_seq)) // 3)]
        valid_tokens_array, valid_valid_len = Vocab.build_array_nmt([words], self.vocab, num_steps=102)
        valid_x = valid_tokens_array.to(device)
        outputs = self.al_net(valid_x)
        outputs = sigmoid(outputs)
        label = torch.max(outputs.data, dim=1)[1]
        proba = torch.max(outputs.data, dim=1)[0]
        return int(label.numpy()), float(proba.numpy())


class sORF:
    def __init__(self, rna_id, sorf_num, sorf_seq, begin_index, end_index, sorf_coding, protein, proba):
        self.rna_id = rna_id
        self.sorf_num = sorf_num
        self.sorf_seq = sorf_seq
        self.start_code_site = begin_index
        self.end_code_site = end_index
        self.sorf_coding = sorf_coding
        self.protein = protein
        self.proba = proba


def Predict_csORFs_by_RNA(input_file, output_file):
    model = Recognizer()
    model.load_train_model(load_path='./checkpoint/', filename='model_weights.pth')
    sorf_list = []
    with open(input_file, 'r') as fa:
        fa_dict = {}
        for line in fa:
            line = line.replace('\n', '')
            if line.startswith('>'):
                seq_name = line[1:]
                fa_dict[seq_name] = ''
            else:
                fa_dict[seq_name] += line.replace('\n', '')
    CountNum = 0
    for item in fa_dict.items():
        rna_id, rna_seq = item[0].split(' ')[0], item[1]
        begin_index_list = [i.start() for i in re.finditer('ATG', rna_seq)]
        end_index_list = [i.start() for i in re.finditer(r'TAA|TGA|TAG', rna_seq)]
        Number = 0
        for i in range(0, 3):
            index_sections = get_index_section(begin_index_list, end_index_list, i)
            for section in index_sections:
                begin_index, end_index = section[0], section[1]
                if end_index - begin_index > 303: continue
                Number += 1
                sorf_seq = rna_seq[begin_index: end_index]
                label, proba = model.predict(sorf_seq)
                sorf_coding = 'coding sORF' if label else 'non-coding sORF'
                protein = sorf_to_protein(sorf_seq) if label else ' '
                sorf_ = sORF(rna_id, "sORF" + str(Number), sorf_seq, begin_index, end_index, sorf_coding, protein,
                             proba)
                sorf_list.append(sorf_)
                CountNum += 1
                if CountNum % 1000 == 0:
                    print('[ {} ] sORFs have been successfully predicted.'.format(CountNum))
    print('[ {} ] sORFs have been successfully predicted.'.format(CountNum))
    print('Prediction End.')
    with open(output_file, 'w+', newline='') as f:
        column = ["RNA_ID", "sORF_ID", "sORF_seq", "start_at", "end_at", "protein", "classification", "probability"]
        writer = csv.writer(f)
        writer.writerow(column)
        for sorf_list_each in sorf_list:
            writer.writerow([sorf_list_each.rna_id, sorf_list_each.sorf_num, sorf_list_each.sorf_seq,
                             sorf_list_each.start_code_site, sorf_list_each.end_code_site,
                             sorf_list_each.protein, sorf_list_each.sorf_coding, sorf_list_each.proba])


def Predict_csORFs_by_sORF(input_file, output_file):
    model = Recognizer()
    model.load_train_model(load_path='./checkpoint/', filename='model_weights.pth')
    sorfs = []
    sorf_list = []
    Number = 0
    with open(input_file, 'r') as fa:
        for line in fa:
            line = line.replace('\n', '')
            sorfs.append(line)
    for sorf in sorfs:
        Number += 1
        label, proba = model.predict(sorf)
        sorf_coding = 'coding sORF' if label else 'non-coding sORF'
        protein = sorf_to_protein(sorf) if label else ' '
        sorf_ = sORF("", "sORF" + str(Number), sorf, -1, -1, sorf_coding, protein, proba)
        sorf_list.append(sorf_)
        if Number % 1000 == 0:
            print('[ {} ] sORFs have been successfully predicted.'.format(Number))
    print('[ {} ] sORFs have been successfully predicted.'.format(Number))
    print('Prediction End.')
    with open(output_file, 'w+', newline='') as f:
        column = ["sORF_ID", "sORF_seq", "protein", "classification", "probability"]
        writer = csv.writer(f)
        writer.writerow(column)
        for sorf_list_each in sorf_list:
            writer.writerow([sorf_list_each.sorf_num, sorf_list_each.sorf_seq,
                             sorf_list_each.protein, sorf_list_each.sorf_coding, sorf_list_each.proba])

if __name__ == '__main__':
    Predict_csORFs_by_RNA(input_file="./input/test.fasta", output_file="./output/test_RNA_sORFs_result.csv")
    pass