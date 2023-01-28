import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc_encoder = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.fc_decoder = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.v = nn.Linear(args.hidden_size, 1, bias=False)

        # self.embedding_dim = args.embedding_dim

    def forward(self, decoder_outputs, encoder_outputs, mask):
        # print(encoder_outputs.shape, self.embedding_dim)
        energy1 = self.fc_encoder(encoder_outputs).unsqueeze(1)
        energy2 = self.fc_decoder(decoder_outputs).unsqueeze(2)
        energy = torch.tanh(energy1 + energy2)
        attention = self.v(energy).squeeze(-1)
        attention = attention.masked_fill(mask.unsqueeze(1), float("-inf"))

        # batch_size, tgt_len, src_len
        return F.softmax(attention, dim=-1)


class Encoder(nn.Module):
    def __init__(self, args, num_layers=4, dropout_rate=0.3, pad_token_id=0):
        super(Encoder, self).__init__()

        self.dropout_rate = dropout_rate
        self.sou_len = args.sou_len
        self.embedding_dim = args.embedding_dim
        self.hidden_size = args.hidden_size
        self.pad_token_id = pad_token_id
        self.num_layers = num_layers

        self.word_embedding = nn.Embedding(num_embeddings=args.vocab_size + args.sou_len, embedding_dim=args.embedding_dim, padding_idx=pad_token_id)

        self.int_gru = nn.GRU(input_size=args.embedding_dim, hidden_size=args.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=False)

        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=True)


    def process_int(self, input_nltxt):
        txt_mask = input_nltxt.gt(0)
        lengths = txt_mask.sum(dim=-1)
        lengths = lengths.masked_fill(lengths == 0, 1)
        word_embedding = self.word_embedding(input_nltxt)
        txtEm = word_embedding

        packed_txtEm = nn.utils.rnn.pack_padded_sequence(input=txtEm, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.int_gru(packed_txtEm)
        pad_packed_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, padding_value=0, total_length=self.sou_len)
        hidden = torch.tanh(self.fc_hidden(hidden))

        return pad_packed_output, hidden

    def forward(self, input_token):
        input_embedding, hidden = self.process_int(input_token)
        return input_embedding, hidden


class Decoder(nn.Module):
    def __init__(self, args, num_layers=4, dropout_rate=0.3, pad_token_id=0):
        super().__init__()
        self.embedding_dim = args.embedding_dim
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.tar_len = args.tar_len
        self.sou_len = args.sou_len
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        self.word_embedding = nn.Embedding(num_embeddings=args.vocab_size + args.sou_len, embedding_dim=args.embedding_dim, padding_idx=pad_token_id)
        self.rnn = nn.GRU(input_size=args.embedding_dim, hidden_size=args.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=False)

        self.attention = Attention(args)

        self.fc_out1 = nn.Linear(self.hidden_size, self.vocab_size + args.sou_len, bias=True)
        self.fc_out2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.fc_gate1 = nn.Linear(self.hidden_size, 1, bias=True)
        self.fc_gate2 = nn.Linear(self.hidden_size, 1, bias=True)
        self.fc_gate3 = nn.Linear(self.embedding_dim, 1, bias=True)

    def process_out(self, input_nltxt, txtEm, hidden):
        txt_mask = input_nltxt.gt(0)
        lengths = txt_mask.sum(dim=-1)
        lengths = lengths.masked_fill(lengths == 0, 1)

        packed_txtEm = nn.utils.rnn.pack_padded_sequence(input=txtEm, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_txtEm, hidden)
        pad_packed_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, padding_value=0, total_length=self.tar_len)
        return pad_packed_output

    def forward(self, sou, tar, hidden, encoder_outputs, extended_length):
        cur_device = sou.device
        tar_embedding = self.word_embedding(tar)
        output = self.process_out(tar, tar_embedding, hidden)
        atten = self.attention(output, encoder_outputs, sou == 0)  # batch_size, tgt_len, src_len
        weighted = torch.bmm(atten, encoder_outputs) 
        all_output = torch.cat((weighted, output), dim=-1)
        output_gen = self.fc_out1(self.fc_out2(all_output))
        extended_mask = torch.ones_like(output_gen, device=cur_device)
        extended_length = extended_length.cpu().numpy().tolist()
        for i in range(len(extended_length)):
            extended_mask[i, :, self.vocab_size + extended_length[i]:] = 0
        output_gen = output_gen.masked_fill(extended_mask == 0, -1e9)
        output_gen = F.softmax(output_gen, dim=-1)

        cur_batch = sou.shape[0]
        
        one_hot_i0 = torch.arange(0, cur_batch, dtype=torch.long, device=cur_device).unsqueeze(1).repeat(1, self.sou_len).view(-1)
        one_hot_i1 = torch.arange(0, self.sou_len, dtype=torch.long, device=cur_device).repeat(cur_batch)
        one_hot_i2 = sou.view(-1)
        one_hot_i = torch.stack([one_hot_i0, one_hot_i1, one_hot_i2])

        one_hot_v = torch.ones([self.sou_len * cur_batch], dtype=torch.float, device=cur_device)
        one_hot = torch.sparse_coo_tensor(one_hot_i, one_hot_v, [cur_batch, self.sou_len, self.vocab_size + self.sou_len])
        output_copy = torch.bmm(one_hot.transpose(1, 2), atten.transpose(1, 2)).transpose(1, 2)

        gate = torch.sigmoid(self.fc_gate1(weighted) + self.fc_gate2(output) + self.fc_gate3(tar_embedding))
        output = (1 - gate) * output_gen + gate * output_copy

        return output, atten
