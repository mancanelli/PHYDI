import math
import torch
import torch.nn as nn
from models.real_model.encoder import EncoderLayer
from models.real_model.decoder import DecoderLayer
from models.real_embedding.positional_encoding import PositionalEncoding
from models.real_embedding.embedding import TokenEmbedding

class Transformer(nn.Module):
    def __init__(self, nhead, num_encoder_layers, num_decoder_layers,
                 emb_size, src_vocab_size, tgt_vocab_size, dim_feedforward=512, 
                 dropout=0.1, ln_vers="pre", rezero=False):
        super(Transformer, self).__init__()

        encoder_layer = EncoderLayer(d_model=emb_size, nhead=nhead, 
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout, rezero=rezero, 
                                    ln_vers=ln_vers)
        decoder_layer = DecoderLayer(d_model=emb_size, nhead=nhead, 
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout, rezero=rezero, 
                                    ln_vers=ln_vers)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)

        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask,
                tgt_padding_mask, memory_key_padding_mask):
        
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)



class EncTransformer(nn.Module):
    def __init__(self, nhead, num_encoder_layers,
                 emb_size, src_vocab_size, dim_feedforward=512, 
                 dropout=0.1, ln_vers="pre", rezero=False):
        super(EncTransformer, self).__init__()

        self.emb_size = emb_size

        encoder_layer = EncoderLayer(d_model=emb_size, nhead=nhead, 
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout, rezero=rezero, 
                                    ln_vers=ln_vers)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.generator = nn.Linear(emb_size, src_vocab_size)
        
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        
        self._reset_parameters()
        self._init_weights()
    
    def _init_weights(self):
        initrange = 0.1
        self.generator.bias.data.zero_()
        self.generator.weight.data.uniform_(-initrange, initrange)
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def square_subsequent_mask(self, size):
        mask = torch.tril(torch.ones(size, size))
        mask = mask.masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask

    def forward(self, src):
        src_mask = self.square_subsequent_mask(len(src)).to(self.device)
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        
        outs = self.transformer_encoder(src_emb, src_mask)
        return self.generator(outs)
