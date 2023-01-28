import torch
import torch.nn.functional as F
from torch import nn

from encoder_decoder import Decoder, Encoder


class TransModel(nn.Module):
    def __init__(self, args, times):
        super(TransModel, self).__init__()
        self.times = times
        self.mark_embedding_dim = args.mark_embedding_dim
        self.word_embedding_dim = args.word_embedding_dim
        self.vocab_size = args.vocab_size
        self.sou_len = args.sou_len
        self.att_len = args.att_len
        
        self.encoder = Encoder(args, pad_token_id=0, times=self.times)
        self.decoder = Decoder(args, pad_token_id=0, times=self.times)

  
    def forward(self, sou, tar, attr, mark, stage = 'train'):
        sou_embedding, hidden = self.encoder(sou, attr, mark)
        tar_output, _ = self.decoder(sou, tar, hidden, sou_embedding)
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
        elif stage == 'dev' or stage == 'test':
            return torch.argmax(tar_output, dim=-1)
