import torch
import torch.nn.functional as F
from torch import nn

from encoder_decoder import Decoder, Encoder


class TransModel(nn.Module):
    def __init__(self, args):
        super(TransModel, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.sou_len = args.sou_len
        
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    # src tgt attr mark
    def forward(self, sou, tar, extended_length, stage = 'train'):
        sou_embedding, hidden = self.encoder(sou)
        tar_output, _ = self.decoder(sou, tar, hidden, sou_embedding, extended_length)
        tar_output = torch.log(tar_output.clamp(min=1e-10))

        pads = torch.zeros(tar.size(0),1) 
        label = torch.cat([tar, pads.cuda(sou.device)], dim=-1)
        label = label[:,1:]
        label = label.long()
        mask = label != 0

        loss = F.nll_loss(tar_output.view(-1, self.vocab_size + self.sou_len), label.contiguous().view(-1), reduction = 'none')
        loss = loss.masked_fill(mask.view(-1)==False, 0)

        if stage == 'train':
            return loss.sum(), mask.sum()
        elif stage == 'dev' or stage == 'test':
            return torch.argmax(tar_output, dim=-1)
