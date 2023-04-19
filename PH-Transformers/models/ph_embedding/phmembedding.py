import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PHMEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, n, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PHMEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq

        if _weight is None:
            self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
            self.S = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, num_embeddings//n, embedding_dim//n))))
            self.weight = torch.zeros((num_embeddings, embedding_dim))
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = nn.Parameter(_weight)

        self.sparse = sparse

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self):
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
    
    def kronecker_product1(self, a, b):
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        out = res.reshape(siz0 + siz1)
        return out
    
    def forward(self, input):
        self.weight = torch.sum(self.kronecker_product1(self.A, self.S), dim=0)

        return F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


class PHMTokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, n):
        super(PHMTokenEmbedding, self).__init__()
        
        self.embedding = PHMEmbedding(vocab_size, emb_size, n)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
