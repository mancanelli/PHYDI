import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ph_layers.phmlinear import PHMLinear
from models.ph_layers.phmattention import PHMMultiheadAttention

class EncoderLayer(nn.Module):
    def __init__(self, n, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation="relu", ln_vers="pre", layer_norm_eps=1e-5, rezero=False):
        super(EncoderLayer, self).__init__()

        self.self_attn = PHMMultiheadAttention(n, d_model, nhead, dropout=dropout)

        self.linear1 = PHMLinear(n, d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = PHMLinear(n, dim_feedforward, d_model)

        self.rezero = rezero
        self.ln_vers = ln_vers

        if self.rezero:
            self.res_weight1 = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.res_weight2 = nn.Parameter(torch.zeros(1), requires_grad=True)
        if self.ln_vers != None:
            self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
    
    def _get_activation_fn(self, activation):
        if activation == "gelu":
            return F.gelu
        else:
            return F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # attention
        src2 = src
        if self.ln_vers == "pre":
            src2 = self.ln1(src2)

        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.dropout1(src2)

        if self.rezero:
            src = src + self.res_weight1 * src2
        elif self.ln_vers == "gpt2":
            src = src + self.ln1(src2)
        else:
            src = src + src2

        if self.ln_vers == "post":
            src = self.ln1(src)
        
        # feedforward
        src2 = src
        if self.ln_vers == "pre":
            src2 = self.ln2(src2)
        
        src2 = self.activation(self.linear1(src2))
        src2 = self.dropout(src2)
        src2 = self.linear2(src2)
        src2 = self.dropout2(src2)

        if self.rezero:
            src = src + self.res_weight2 * src2
        elif self.ln_vers == "gpt2":
            src = src + self.ln2(src2)
        else:
            src = src + src2

        if self.ln_vers == "post":
            src = self.ln2(src)

        return src
