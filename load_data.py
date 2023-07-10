import torch
from Vocab import load_array


batch_size = 64
src = torch.load("../data/p_n_pair_src.pth")
tgt = torch.load("../data/p_n_pair_tgt.pth")
data_arrays = (src, tgt)
ps_data_iter = load_array(data_arrays, batch_size)
src = torch.load("../data/n_p_pair_src.pth")
tgt = torch.load("../data/n_p_pair_tgt.pth")
ng_data_arrays = (src, tgt)
ng_data_iter = load_array(ng_data_arrays, batch_size)
