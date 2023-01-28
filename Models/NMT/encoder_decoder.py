import torch.nn as nn
import torch.nn.functional as F
import torch
import math



class BahdanauAttention(nn.Module):
   
    
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        
        query_size = hidden_size 
        key_size = 2 * hidden_size 

        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
      
        self.alphas = None
        
    def forward(self, query, proj_key, value, mask):

        query = self.query_layer(query)
    
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
  

    
        scores.data.masked_fill_(mask.unsqueeze(1) == 0, -float('inf'))
  
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas        
        

        context = torch.bmm(alphas, value)

        return context, alphas



class Encoder(nn.Module):
    
    def __init__(self, args, num_layers=3, pad_token_id=0, dropout=0.):
        super(Encoder, self).__init__()

        self.dropout = dropout
        self.sou_len = args.sou_len
        self.embedding_dim = args.embedding_dim
        self.hidden_size = args.hidden_size
        self.pad_token_id = pad_token_id
        self.num_layers = num_layers

        self.word_embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.embedding_dim, padding_idx=pad_token_id)

        self.int_gru = nn.GRU(input_size=args.embedding_dim, hidden_size=args.hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=self.dropout)

    def process_int(self, input_nltxt):
        cur_batch = input_nltxt.size(0)
        txt_mask = input_nltxt.gt(0)
        lengths = txt_mask.sum(dim=-1)
        lengths = lengths.masked_fill(lengths == 0, 1)
        word_embedding = self.word_embedding(input_nltxt)
        txtEm = word_embedding

        packed_txtEm = nn.utils.rnn.pack_padded_sequence(input=txtEm, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.int_gru(packed_txtEm)
  
        pad_packed_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, padding_value=0, total_length=self.sou_len)
        hidden = hidden.view(self.num_layers, 2, cur_batch, self.hidden_size)
        hidden = hidden.transpose(1, 2)
        hidden = hidden.reshape(self.num_layers, cur_batch, self.hidden_size*2)

        return pad_packed_output, hidden

    def forward(self, input_token):

        input_embedding, hidden = self.process_int(input_token)
        
     
        return input_embedding, hidden

class Decoder(nn.Module):

    
    def __init__(self, args, num_layers=3, pad_token_id=0, dropout=0.5, bridge=True):
        super(Decoder, self).__init__()
        
        self.embedding_dim = args.embedding_dim
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.tar_len = args.tar_len
        self.num_layers = num_layers
        self.dropout = dropout

        self.word_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim, padding_idx=pad_token_id) 
        self.attention = BahdanauAttention(self.hidden_size)
        self.rnn = nn.GRU(self.embedding_dim + 2*self.hidden_size, self.hidden_size, num_layers, batch_first=True, dropout=dropout)
                 

        self.bridge = nn.Linear(2*self.hidden_size, self.hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(self.hidden_size + 2*self.hidden_size + self.embedding_dim,
                                          self.hidden_size, bias=False)
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def forward_step(self, prev_embed, encoder_outputs, src_mask, proj_key, hidden):

        query = hidden[-1].unsqueeze(1)  
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_outputs, mask=src_mask)


        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)
 
        pre_output = self.fc_out(pre_output)
        pre_output = F.softmax(pre_output, dim=-1)
        return output, hidden, pre_output
    
    def forward(self, sou, tar, encoder_outputs, encoder_hiddens, max_len=None):
      

        src_mask = sou != 0
        trg_embed = self.word_embedding(tar)
                    

     
        hidden = torch.tanh(self.bridge(encoder_hiddens))          
        
     
        proj_key = self.attention.key_layer(encoder_outputs)
    
        decoder_states = []
        pre_output_vectors = []
        
        if max_len is None:
            max_len = self.tar_len

        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
              prev_embed, encoder_outputs, src_mask, proj_key, hidden)
            assert (output == hidden[-1].unsqueeze(1)).all()
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        
        return decoder_states, hidden, pre_output_vectors 
  
