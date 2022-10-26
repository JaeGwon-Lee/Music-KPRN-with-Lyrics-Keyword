import torch
import torch.nn as nn
import torch.nn.functional as F



# KPRN 구조
class KPRN(nn.Module):
    
    def __init__(self, e_emb_dim, t_emb_dim, r_emb_dim, hidden_dim, e_vocab_size,
                 t_vocab_size, r_vocab_size, tagset_size, no_rel, rnn_type):
        
        super(KPRN, self).__init__()
        self.hidden_dim = hidden_dim
        self.entity_embeddings = nn.Embedding(e_vocab_size, e_emb_dim)
        self.type_embeddings = nn.Embedding(t_vocab_size, t_emb_dim)
        self.rel_embeddings = nn.Embedding(r_vocab_size, r_emb_dim)

        # RNN Layer
        # input : 임베딩된 텍스트 / output : hidden states
        if no_rel:
            if rnn_type == 'rnn' :
                self.rnn = nn.RNN(e_emb_dim + t_emb_dim, hidden_dim)
            elif rnn_type == 'gru' :
                self.rnn = nn.GRU(e_emb_dim + t_emb_dim, hidden_dim)
            else :
                self.rnn = nn.LSTM(e_emb_dim + t_emb_dim, hidden_dim)
        else:
            if rnn_type == 'rnn' :
                self.rnn = nn.RNN(e_emb_dim + t_emb_dim + r_emb_dim, hidden_dim)
            elif rnn_type == 'gru' :
                self.rnn = nn.GRU(e_emb_dim + t_emb_dim + r_emb_dim, hidden_dim)
            else :
                self.rnn = nn.LSTM(e_emb_dim + t_emb_dim + r_emb_dim, hidden_dim)
        
        # Pooling Layer
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, tagset_size)
    
    
    
    # Forward 함수
    def forward(self, paths, path_lengths, no_rel, rnn_type):

        # path 구조 변경 => 1st row : entities / 2nd row : types / 3rd row : relations
        t_paths = torch.transpose(paths, 1, 2)

        # path 임베딩
        entity_embed = self.entity_embeddings(t_paths[:,0,:])
        type_embed = self.type_embeddings(t_paths[:,1,:])
        if no_rel:
            triplet_embed = torch.cat((entity_embed, type_embed), 2)
        else:
            rel_embed = self.rel_embeddings(t_paths[:,2,:])
            triplet_embed = torch.cat((entity_embed, type_embed, rel_embed), 2)

        # (input size x batch_size x embedding dim)로 구조 변경
        batch_sec_embed = torch.transpose(triplet_embed, 0 , 1)

        # 패딩
        packed_embed = nn.utils.rnn.pack_padded_sequence(batch_sec_embed, path_lengths.to('cpu'))

        # RNN Layer
        if rnn_type == 'rnn' or rnn_type == 'gru' :
            packed_out, last_out = self.rnn(packed_embed)    # RNN, GRU의 output 구조
        else :
            packed_out, (last_out, _) = self.rnn(packed_embed)    # LSTM의 output 구조

        # Pooling Layer
        tag_scores = self.linear2(F.relu(self.linear1(last_out[-1])))

        return tag_scores
    
    
    
    # KPRN 유사도 함수
    def weighted_pooling(self, path_scores, gamma=1):

        exp_weighted = torch.exp(torch.div(path_scores, gamma))
        sum_exp = torch.sum(exp_weighted, dim=0)
        
        return torch.log(sum_exp)