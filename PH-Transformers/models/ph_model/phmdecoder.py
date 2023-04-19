import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ph_layers.phmlinear import PHMLinear
from models.ph_layers.phmattention import PHMMultiheadAttention

class DecoderLayer(nn.Module):
    def __init__(self, n, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation="relu", ln_vers="pre", layer_norm_eps=1e-5, rezero=False):
        super(DecoderLayer, self).__init__()

        self.self_attn = PHMMultiheadAttention(n, d_model, nhead, dropout=dropout)
        self.multihead_attn = PHMMultiheadAttention(n, d_model, nhead, dropout=dropout)

        self.linear1 = PHMLinear(n, d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = PHMLinear(n, dim_feedforward, d_model)

        self.rezero = rezero
        self.ln_vers = ln_vers

        if self.rezero:
            self.res_weight1 = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.res_weight2 = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.res_weight3 = nn.Parameter(torch.zeros(1), requires_grad=True)
        if self.ln_vers != None:
            self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.ln3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
    
    def _get_activation_fn(self, activation):
        if activation == "gelu":
            return F.gelu
        else:
            return F.relu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # attention
        if self.ln_vers == "pre":
            tgt = self.ln1(tgt)

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt2 = self.dropout1(tgt2)

        if self.rezero:
            tgt = tgt + self.res_weight1 * tgt2
        elif self.ln_vers == "gpt2":
            tgt = tgt + self.ln1(tgt2)
        else:
            tgt = tgt + tgt2

        if self.ln_vers == "post":
            tgt = self.ln1(tgt)
        
        # attention
        if self.ln_vers == "pre":
            tgt = self.ln2(tgt)
        
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = self.dropout2(tgt2)

        if self.rezero:
            tgt = tgt + self.res_weight2 * tgt2
        elif self.ln_vers == "gpt2":
            tgt = tgt + self.ln2(tgt2)
        else:
            tgt = tgt + tgt2

        if self.ln_vers == "post":
            tgt = self.ln2(tgt)
        
        # feedforward
        if self.ln_vers == "pre":
            tgt = self.ln3(tgt)
        
        tgt2 = self.activation(self.linear1(tgt))
        tgt2 = self.dropout(tgt2)
        tgt2 = self.linear2(tgt2)
        tgt2 = self.dropout3(tgt2)

        if self.rezero:
            tgt = tgt + self.res_weight3 * tgt2
        elif self.ln_vers == "gpt2":
            tgt = tgt + self.ln3(tgt2)
        else:
            tgt = tgt + tgt2

        if self.ln_vers == "post":
            tgt = self.ln3(tgt)

        return tgt