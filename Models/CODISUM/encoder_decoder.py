import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc_encoder = nn.Linear(args.embedding_dim * 2, args.embedding_dim, bias=False)
        self.fc_decoder = nn.Linear(args.embedding_dim, args.embedding_dim, bias=False)
        self.v = nn.Linear(args.embedding_dim, 1, bias=False)



    def forward(self, decoder_outputs, encoder_outputs, mask):

        energy1 = self.fc_encoder(encoder_outputs).unsqueeze(1)
        energy2 = self.fc_decoder(decoder_outputs).unsqueeze(2)
        energy = torch.tanh(energy1 + energy2)
        attention = self.v(energy).squeeze(-1)
        attention = attention.masked_fill(mask.unsqueeze(1), float("-inf"))


        return F.softmax(attention, dim=-1)


class Encoder(nn.Module):
    def __init__(self, args, pad_token_id, times):
        super(Encoder, self).__init__()
        self.times = times

        self.dropout_rate = args.dropout_rate
        self.sou_len = args.sou_len
        self.att_len = args.att_len
        self.mark_embedding_dim = args.mark_embedding_dim
        self.word_embedding_dim = args.word_embedding_dim
        self.embedding_dim = args.embedding_dim
        self.pad_token_id = pad_token_id
        self.num_layers = 3

        self.word_embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.word_embedding_dim, padding_idx=pad_token_id)
        self.mark_embedding = nn.Embedding(num_embeddings=4, embedding_dim=self.mark_embedding_dim, padding_idx=pad_token_id)

        self.int_gru = nn.GRU(input_size=args.word_embedding_dim + args.mark_embedding_dim, hidden_size=args.embedding_dim // 2, num_layers=3, batch_first=True, bidirectional=True)
        self.sub_gru = nn.GRU(input_size=args.word_embedding_dim, hidden_size=args.embedding_dim // 2, num_layers=self.num_layers, batch_first=True, bidirectional=True)

        self.fc_hidden = nn.Linear(self.embedding_dim, self.embedding_dim)

    def process_int(self, input_nltxt, mark):
        cur_batch = input_nltxt.size(0)
        txt_mask = input_nltxt.gt(0)
        lengths = txt_mask.sum(dim=-1)
        lengths = lengths.masked_fill(lengths == 0, 1)
        word_embedding = self.word_embedding(input_nltxt)
        mark_embedding = self.mark_embedding(mark)
        txtEm = torch.cat((mark_embedding, word_embedding), dim=-1)

        packed_txtEm = nn.utils.rnn.pack_padded_sequence(input=txtEm, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.int_gru(packed_txtEm)
    
        pad_packed_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, padding_value=0, total_length=self.sou_len)
        hidden = hidden.view(self.num_layers, 2, cur_batch, self.embedding_dim // 2)
        hidden = hidden.transpose(1, 2)
        hidden = hidden.reshape(self.num_layers, cur_batch, self.embedding_dim)
    

        return pad_packed_output, hidden


    def process_txt(self, input_nltxt):
        cur_batch = input_nltxt.size(0)  
        input_nltxt = input_nltxt.view(-1, self.att_len)  
        txt_mask = input_nltxt.gt(0)
        lengths_old = txt_mask.sum(dim=-1)
        lengths = lengths_old.masked_fill(lengths_old == 0, 1)

        txtEm = self.word_embedding(input_nltxt)  

        packed_txtEm = nn.utils.rnn.pack_padded_sequence(input=txtEm, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.sub_gru(packed_txtEm)  
        pad_packed_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True,
        padding_value=0, total_length=self.att_len)
     
        txtEm = pad_packed_output[range(lengths.size(0)), lengths - 1] 
        txtEm = torch.masked_fill(txtEm, lengths_old.unsqueeze(-1) == 0, 0)  
        txtEm = txtEm.view(-1, self.sou_len, self.embedding_dim) 
      
        return txtEm

    

    def forward(self, input_token, attr, mark):
        input_embedding, hidden = self.process_int(input_token, mark)
        att_embedding = self.process_txt(attr)
        intput_embedding = torch.cat((input_embedding, att_embedding), dim=-1)
        return intput_embedding, hidden


class Decoder(nn.Module):
    def __init__(self, args, pad_token_id, times):
        super().__init__()
        self.word_embedding_dim = args.word_embedding_dim
        self.embedding_dim = args.embedding_dim
        self.vocab_size = args.vocab_size
        self.tar_len = args.tar_len
        self.sou_len = args.sou_len

        self.word_embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.word_embedding_dim, padding_idx=pad_token_id)
        self.rnn = nn.GRU(input_size=args.word_embedding_dim, hidden_size=args.embedding_dim, num_layers=3, batch_first=True, bidirectional=False)

        self.attention = Attention(args)

        self.fc_out = nn.Linear(self.embedding_dim * 3, self.vocab_size)
        self.fc_gate = nn.Linear(self.embedding_dim * 3, 1)

    def process_out(self, input_nltxt, hidden):
        txt_mask = input_nltxt.gt(0)
        lengths = txt_mask.sum(dim=-1)
        lengths = lengths.masked_fill(lengths == 0, 1)

        txtEm = self.word_embedding(input_nltxt)
        packed_txtEm = nn.utils.rnn.pack_padded_sequence(input=txtEm, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_txtEm, hidden)
        pad_packed_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, padding_value=0, total_length=self.tar_len)
        return pad_packed_output

    def forward(self, sou, tar, hidden, encoder_outputs):
        output = self.process_out(tar, hidden)
        atten = self.attention(output, encoder_outputs, sou == 0)  
        weighted = torch.bmm(atten, encoder_outputs)
        all_output = torch.cat((weighted, output), dim=-1)
        output_gen = self.fc_out(all_output)
        output_gen = F.softmax(output_gen, dim=-1)

        cur_batch = sou.shape[0]
        cur_device = sou.device

        one_hot_i0 = torch.arange(0, cur_batch, dtype=torch.long, device=cur_device).unsqueeze(1).repeat(1, self.sou_len).view(-1)
        one_hot_i1 = torch.arange(0, self.sou_len, dtype=torch.long, device=cur_device).repeat(cur_batch)
        one_hot_i2 = sou.view(-1)
        one_hot_i = torch.stack([one_hot_i0, one_hot_i1, one_hot_i2])

        one_hot_v = torch.ones([self.sou_len * cur_batch], dtype=torch.float, device=cur_device)
        one_hot = torch.sparse_coo_tensor(one_hot_i, one_hot_v, [cur_batch, self.sou_len, self.vocab_size])
        output_copy = torch.bmm(one_hot.transpose(1, 2), atten.transpose(1, 2)).transpose(1, 2)

   

        gate = torch.sigmoid(self.fc_gate(all_output))
        output = (1 - gate) * output_gen + gate * output_copy

        return output, atten
