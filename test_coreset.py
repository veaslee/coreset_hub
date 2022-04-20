import torch
import numpy as np
from utils import test, load_coreset_index
from load_data import load_cifar_coreset_complement
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--sample_size_frac', type=float, default=1.0, help='dataset')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--load_path', type=str, default='../coreset_models/', help='iteration index')
args = parser.parse_args()

print('--------args----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))
print('--------args----------\n')

dataset = args.dataset
sample_size_frac = args.sample_size_frac
batch_size = args.batch_size
load_path = args.load_path
sample_size = int(sample_size_frac*50000)
corese_index_path = '../run/' + dataset + '/' + str(sample_size_frac) + '/' + str(sample_size) + '_selected.index'
coreset_index = load_coreset_index(corese_index_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()


# Compute coreset complement risk
for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    coreset_complement = load_cifar_coreset_complement(dataset, batch_size, coreset_index, frac)
    risk_list = []
    for index in range(10):
        net = torch.load(load_path + 'resnet_' + dataset + '_sample_size' + str(sample_size) + '_frac_' + str(frac) + '_index_' + str(index+1))
        _, acc = test(net, coreset_complement)
        risk_list.append(1-acc)


    print('frac: ' + str(frac))
    print('coreset complement risk: ' + str(np.mean(risk_list)))
