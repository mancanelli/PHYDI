import os
import argparse
import wandb
import torch

from training import Trainer
from utils.dataloaders import DatasetHandler1, DatasetHandler2
from utils.readFile import readFile
from models.ph_model.phmtransformer import EncTransformer, Transformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1656079)
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--n_workers', default=1)
    
    parser.add_argument('--max_tokens', type=int, default=1000)
    parser.add_argument('--n', type=int, default=4, help="n parameter for PHM layer")
    parser.add_argument('--optim', type=str, default="SGD")
    parser.add_argument('--scheduler', type=str, default="cosine")
    parser.add_argument('--train_dir', type=str, default='./data/', help="folder containg training data")
    
    parser.add_argument('--task', type=int, default=1)
    parser.add_argument('--rezero', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--betas', default=(0.0, 0.9))
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--ln_vers', type=str, default="pre", help="layernorm version")
    parser.add_argument('--num_encoder_layers', type=int, default=12, help="number of layers")
    parser.add_argument('--num_decoder_layers', type=int, default=12, help="number of layers")
    parser.add_argument('--emb_size', type=int, default=128, help="embedding dimension")
    parser.add_argument('--nhid', type=int, default=256, help="dimension of the feedforward network model")
    parser.add_argument('--nhead', type=int, default=8, help="number of heads in self attention")

    parser.add_argument('--EpochCheckpoints', type=bool, default=True, help='Save model every epoch. If set to False the model will be saved only at the end')
    
    parser.add_argument('--TextArgs', type=str, default='TrainingArguments.txt', help='Path to text with training settings')
    parse_list = readFile(parser.parse_args().TextArgs)
    
    opt = parser.parse_args(parse_list)
    

    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and opt.cuda else "cpu")

    dh = DatasetHandler1(opt.max_tokens, opt.batch_size, opt.eval_batch_size, device)
    train_data, val_data = dh.getData()

    src_vocab_size = len(dh.vocab)

    net = EncTransformer(n=opt.n, src_vocab_size=src_vocab_size, emb_size=opt.emb_size, nhead=opt.nhead, 
                        dim_feedforward=opt.nhid, num_encoder_layers=opt.num_encoder_layers, dropout=opt.dropout, 
                        ln_vers=opt.ln_vers, rezero=opt.rezero)
    net.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(net.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.9)
    
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of parameters:', params)


    wandb.init(project="rezero-phtransformer", dir="./PH-Transformers/")
    config = wandb.config
    wandb.watch(net)
    wandb.config.update(opt)
    #wandb.config.update({"params": params})


    checkpoint_folder = './PH-Transformers/checkpoints/'
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    

    trainer = Trainer(net, optimizer, scheduler, criterion, epochs=opt.epochs, 
                      device=device, checkpoint_folder=checkpoint_folder,
                      lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    
    trainer.train1(train_data, val_data, src_vocab_size)