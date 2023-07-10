import csv
import re
import pandas as pd


# 起始密码子 ATG 终止密码子 TAA TGA TAG


def get_index_section(begin_index_list, end_index_list, i):
    begin_index_list_i, end_index_list_i = [], []
    for x in begin_index_list:
        if x % 3 == i:
            begin_index_list_i.append(x)
    for y in end_index_list:
        if y % 3 == i:
            end_index_list_i.append(y)
    b_i, e_i = 0, 0
    index_sections = []
    now_end_index = -1
    while b_i < len(begin_index_list_i) and e_i < len(end_index_list_i):
        if begin_index_list_i[b_i] < now_end_index:
            b_i = b_i + 1
        elif begin_index_list_i[b_i] < end_index_list_i[e_i]:
            index_sections.append((begin_index_list_i[b_i], end_index_list_i[e_i] + 3))
            now_end_index = end_index_list_i[e_i]
            b_i, e_i = b_i + 1, e_i + 1
        else:
            e_i = e_i + 1
    return index_sections


def sorf_to_protein(sorf_seq):
    protein_seq = ""
    protein_dict = {'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
                    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*', 'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
                    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L', 'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
                    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
                    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M', 'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
                    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
                    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V', 'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
                    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'}
    i, length = 0, len(sorf_seq) - 3
    while i < length:
        protein_seq += protein_dict[sorf_seq[i:i + 3]]
        i += 3
    # print('protein_seq = ')
    # print(protein_seq)
    return protein_seq


class sorf:
    def __init__(self, rna_id, rna_name, rna_seq_num, rna_coding, sorf_num, sorf_seq, begin_index, end_index,
                 sorf_coding, protein, bia_forward, bia_backward):
        self.rna_id = rna_id
        self.rna_name = rna_name
        self.rna_seq_num = rna_seq_num
        self.rna_coding = rna_coding
        self.sorf_num = sorf_num
        self.sorf_seq = sorf_seq
        self.start_code_site = begin_index
        self.end_code_site = end_index
        self.sorf_coding = sorf_coding
        self.protein = protein
        self.sorf_length = end_index - begin_index
        self.bia_forward = bia_forward
        self.bia_backward = bia_backward


if __name__ == '__main__':
    rna_datasets = pd.read_csv('./new_datasets.csv')
    sorf_list = []
    k = 0
    for rna_info in rna_datasets.itertuples():
        rna_id = getattr(rna_info, 'rna_id')
        rna_name = getattr(rna_info, 'rna_name')
        rna_seq_num = getattr(rna_info, 'rna_seq_num')
        rna_coding = getattr(rna_info, 'rna_coding')
        rna_seq = getattr(rna_info, 'rna_seq')
        rna_protein = getattr(rna_info, 'rna_protein')
        begin_index_list = [i.start() for i in re.finditer('ATG', rna_seq)]
        end_index_list = [i.start() for i in re.finditer(r'TAA|TGA|TAG', rna_seq)]
        cp_flag = False
        Number = 0
        for i in range(0, 3):
            index_sections = get_index_section(begin_index_list, end_index_list, i)
            for section in index_sections:
                begin_index, end_index = section[0], section[1]
                if end_index - begin_index > 303:
                    continue
                Number += 1
                sorf_seq = rna_seq[begin_index: end_index]
                bia_forward = rna_seq[:begin_index] if begin_index <= 174 else rna_seq[begin_index - 174:begin_index]
                bia_forward_len = len(bia_forward)
                bia_backward = rna_seq[end_index:end_index + 30]
                sorf_coding = False
                protein = sorf_to_protein(sorf_seq)
                if protein == rna_protein:
                    sorf_coding = True
                    cp_flag = True
                sorf_ = sorf(rna_id, rna_name, rna_seq_num, rna_coding, "sorf" + str(Number), sorf_seq, begin_index,
                             end_index, sorf_coding, protein, bia_forward, bia_backward)
                sorf_list.append(sorf_)
    with open('./sorf_datasets_1.csv', 'w+', newline='') as f:
        column = ["rna_id", "rna_name", "rna_seq_num", "rna_coding", "sorf_num", "start_code_site", "end_code_site",
                  "sorf_length", "sorf_seq", "protein", "sorf_coding", "bia_forward", "bia_backward"]
        writer = csv.writer(f)
        writer.writerow(column)
        for sorf_list_each in sorf_list:
            writer.writerow([sorf_list_each.rna_id, sorf_list_each.rna_name, sorf_list_each.rna_seq_num,
                             sorf_list_each.rna_coding, sorf_list_each.sorf_num, sorf_list_each.start_code_site,
                             sorf_list_each.end_code_site, sorf_list_each.sorf_length, sorf_list_each.sorf_seq,
                             sorf_list_each.protein, sorf_list_each.sorf_coding, sorf_list_each.bia_forward,
                             sorf_list_each.bia_backward])
