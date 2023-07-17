import random
import torch
from model.Generator import MyTransformer
import Vocab
from Vocab import VocabClass
from utils.vocab_utils import vocab

types = 'ng'
load_path = './checkpoint/'
file_name = '../data/generator_samples_' + types + '.fa'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dataset_name='positive_sorf.fa'
generated_nums = 1000

class GenerateSamplesClass:
    def __init__(self, data_dir='../data/', embedding_dim=128, batch_size=64, vocab_size=68,
                 token_len=102, hidden_size=256, dropout=0):
        self.batch_size = batch_size
        self.token_len = token_len
        self.Generator = MyTransformer(embedding_dim, batch_size, vocab_size, token_len, hidden_size, dropout)
        self.Generator.to(device)
        self.vocab = vocab
        self.load_model(load_path=load_path)
        self.input_iter = self.load_data(data_dir, input_dataset_name, self.vocab)


    def load_model(self, load_path):
        if len(load_path) == 0:
            print("[*] Checkpoint not found! Starting from scratch.")
            return 0  # file is not there
        Trans_G_weights = torch.load(load_path + "train_G_weights_" + types + ".pth", map_location=device)
        self.Generator.load_state_dict(Trans_G_weights)

    def load_data(self, data_dir, filename, vocab, num_steps=102):
        words_list = Vocab.getWordsList(data_dir + filename)
        tokens_array, _ = Vocab.build_array_nmt(words_list, vocab, num_steps)
        tgt = torch.ones((tokens_array.shape[0], 1))
        arrays = (tokens_array, tgt)
        return Vocab.load_array(arrays, self.batch_size)

    def generator_sample_batch(self, real_data):
        generated_data = torch.IntTensor().to(device)
        for real_token in real_data:
            gen_sample = self.generator_sample_one_seq(real_token, topk_num=2)
            generated_data = torch.cat([generated_data, gen_sample], dim=0)
        return generated_data

    def generator_sample_one_seq(self, real_token, num_steps=102, device=device, topk_num=2):
        enc_X = real_token.reshape(1, -1).to(device)
        dec_X = torch.unsqueeze(torch.tensor([self.vocab['<bos>']], dtype=torch.long, device=device), dim=0)
        for _ in range(num_steps):
            outputs = self.Generator(enc_X, dec_X)
            predict = self.Generator.predictor(outputs)[-1]
            top_word = torch.argmax(predict, dim=1)
            pred = top_word.item()
            if pred == self.vocab['<eos>']:
                break
            elif pred == self.vocab['ATG'] and len(dec_X[0]) == 1:
                y = top_word.reshape((1, -1))
            else:
                values, indices = torch.topk(predict, k=topk_num, dim=1, largest=True, sorted=True)
                indices = indices.reshape(-1)
                y = indices[random.randint(0, topk_num - 1)].reshape((1, -1))
                pred = y.squeeze(dim=0).type(torch.int32).item()
                if pred == self.vocab['<eos>']:
                    break
            dec_X = torch.cat([dec_X, y], dim=1).to(device)
        return torch.cat(
            [dec_X[:, 1:],
             torch.IntTensor([1]).repeat(self.token_len - dec_X[:, 1:].shape[1], 1).reshape(1, -1).to(device)],
            dim=1).to(device)

    def save_generator_samples(self):
        sample_pool = torch.IntTensor().to(device)
        # generate sample to al_model
        d_iter = self.input_iter
        count = 0
        while True:
            for data in d_iter:
                real_data, _ = data
                generate_data = self.generator_sample_batch(real_data)
                sample_pool = torch.cat([sample_pool, generate_data], dim=0)
                count += 1
                print('now count : [{}]'.format(count))
                if count >= generated_nums:
                    break
            if count >= generated_nums:
                break
        with open(file_name, 'w', encoding='utf-8') as file:
            for i in range(len(sample_pool)):
                fake_seq = sample_pool[i].tolist()
                eos = '<pad>'
                fake_txt = ''.join(vocab.to_tokens(fake_seq)).split(eos, 1)[0]
                # print('fake_txt : [ {} ]'.format(fake_txt))
                file.write(fake_txt + '\n')


def main():
    model = GenerateSamplesClass()
    model.save_generator_samples()


if __name__ == '__main__':
    main()
