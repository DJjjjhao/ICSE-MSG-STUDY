import torch
import torch.nn.functional as F
from torch import nn

from encoder_decoder import Decoder, Encoder


class TransModel(nn.Module):
    def __init__(self, args):
        super(TransModel, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.vocab_size = args.vocab_size
        self.sou_len = args.sou_len
        
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)


    def forward(self, sou, tar, stage = 'train'):
        sou_embedding, hidden = self.encoder(sou)
        _, _, tar_output = self.decoder(sou, tar, sou_embedding, hidden)
        tar_output = torch.log(tar_output.clamp(min=1e-10))

        pads = torch.zeros(tar.size(0),1) 
        label = torch.cat([tar, pads.cuda(sou.device)], dim=-1)
        label = label[:,1:]
        label = label.long()
        mask = label != 0

        loss = F.nll_loss(tar_output.view(-1, self.vocab_size), label.contiguous().view(-1), reduction = 'none')
        loss = loss.masked_fill(mask.view(-1)==False, 0)

        if stage == 'train':
            return loss.sum(), mask.sum()
        elif stage == 'dev':
            return torch.argmax(tar_output, dim=-1)
        elif stage == 'test':
            raise
