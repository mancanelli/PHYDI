import io
from collections import Counter
from collections import OrderedDict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchtext.vocab import vocab
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class DatasetHandler1():
    def __init__(self, max_tokens, batch_size, eval_batch_size, device):
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device

        self.tokenizer = None
        self.vocab = None

    def getData(self):
        train_iter = WikiText2(split='train')
        self.tokenizer = get_tokenizer('basic_english')
        
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, train_iter), specials=['<unk>'], max_tokens=self.max_tokens)
        self.vocab.set_default_index(self.vocab['<unk>'])

        train_iter, val_iter, _ = WikiText2()
        train_data = self.data_process(train_iter)
        val_data = self.data_process(val_iter)

        train_data = self.batchify(train_data, self.batch_size)
        val_data = self.batchify(val_data, self.eval_batch_size)

        return train_data, val_data

    def data_process(self, raw_text_iter):
        data = [torch.tensor(self.vocab(self.tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def batchify(self, data, batch_size):
        seq_len = data.size(0) // batch_size
        data = data[:seq_len * batch_size]
        data = data.view(batch_size, seq_len).t().contiguous()
        return data.to(self.device)


class DatasetHandler2():
    def __init__(self, train_filepaths, val_filepaths, max_tokens, batch_size):
        self.train_filepaths = train_filepaths
        self.val_filepaths = val_filepaths
        self.max_tokens = max_tokens
        self.batch_size = batch_size

        self.en_tokenizer = None
        self.modern_vocab = None
        self.original_vocab = None

        self.PAD_IDX = 0
        self.BOS_IDX = 0
        self.EOS_IDX = 0

    def getData(self):
        self.en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

        self.modern_vocab = self.build_vocab(self.train_filepaths[0])
        self.original_vocab = self.build_vocab(self.train_filepaths[1])

        train_data = self.data_process(self.train_filepaths)
        val_data = self.data_process(self.val_filepaths)

        self.PAD_IDX = self.modern_vocab['<pad>']
        self.BOS_IDX = self.modern_vocab['<bos>']
        self.EOS_IDX = self.modern_vocab['<eos>']

        train_data = DataLoader(train_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.generate_batch)
        val_data = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.generate_batch)

        return train_data, val_data

    def build_vocab(self, filepath):
        counter = Counter()

        with io.open(filepath, encoding="utf8") as f:
            for string_ in f:
                counter.update(self.en_tokenizer(string_))
        
        #max_tokens = 12928
        specials=['<unk>', '<pad>', '<bos>', '<eos>']
        
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        ordered_dict = OrderedDict(sorted_by_freq_tuples[:self.max_tokens - len(specials)])
        
        word_vocab = vocab(ordered_dict, min_freq=1, specials=specials)
        word_vocab.set_default_index(word_vocab['<unk>'])
        return word_vocab

    def data_process(self, filepaths):
        raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
        raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
        data = []
        
        for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
            mo_tensor_ = torch.tensor([self.modern_vocab[token] for token in self.en_tokenizer(raw_de.rstrip("\n"))], dtype=torch.long)
            og_tensor_ = torch.tensor([self.original_vocab[token] for token in self.en_tokenizer(raw_en.rstrip("\n"))], dtype=torch.long)
            data.append((mo_tensor_, og_tensor_))
        
        return data

    def generate_batch(self, data_batch):
        de_batch, en_batch = [], []
        
        for (de_item, en_item) in data_batch:
            de_batch.append(torch.cat([torch.tensor([self.BOS_IDX]), de_item, torch.tensor([self.EOS_IDX])], dim=0))
            en_batch.append(torch.cat([torch.tensor([self.BOS_IDX]), en_item, torch.tensor([self.EOS_IDX])], dim=0))
        
        de_batch = pad_sequence(de_batch, padding_value=self.PAD_IDX)
        en_batch = pad_sequence(en_batch, padding_value=self.PAD_IDX)
        return de_batch, en_batch
