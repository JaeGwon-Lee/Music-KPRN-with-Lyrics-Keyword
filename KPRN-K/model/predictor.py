import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import my_collate, sort_batch



class TestInteractionData(Dataset):
    
    def __init__(self, formatted_data):
        self.data = formatted_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



# 데이터 원형으로 변환 (시각화에 사용)
def convert_to_etr(e_to_ix, t_to_ix, r_to_ix, path, length):

    ix_to_t = {v: k for k, v in t_to_ix.items()}
    ix_to_r = {v: k for k, v in r_to_ix.items()}
    ix_to_e = {v: k for k, v in e_to_ix.items()}
    new_path = []
    
    for i,step in enumerate(path):
        if i == length:
            break
        new_path.append([ix_to_e[step[0].item()], ix_to_t[step[1].item()], ix_to_r[step[2].item()]])
    
    return new_path



# 테스트 데이터 predict
def predict(model, formatted_data, batch_size, device, no_rel, gamma, rnn_type):

    prediction_scores = []
    interaction_data = TestInteractionData(formatted_data)
    test_loader = DataLoader(dataset=interaction_data, collate_fn = my_collate, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (interaction_batch, _) in test_loader:
            paths = []
            lengths = []
            inter_ids = []
            for inter_id, interaction_paths in enumerate(interaction_batch):
                for path, length in interaction_paths:
                    paths.append(path)
                    lengths.append(length)
                inter_ids.extend([inter_id for i in range(len(interaction_paths))])

            inter_ids = torch.tensor(inter_ids, dtype = torch.long)
            paths = torch.tensor(paths, dtype=torch.long)
            lengths = torch.tensor(lengths, dtype=torch.long)
            
            s_path_batch, s_inter_ids, s_lengths = sort_batch(paths, inter_ids, lengths)    # path 길이 순 정렬
            tag_scores = model(s_path_batch.to(device), s_lengths.to(device), no_rel, rnn_type)

            start = True
            for i in range(len(interaction_batch)):
                inter_idxs = (s_inter_ids == i).nonzero(as_tuple=False).squeeze(1)
                pooled_score = model.weighted_pooling(tag_scores[inter_idxs], gamma=gamma)

                if start:
                    pooled_scores = pooled_score.unsqueeze(0)
                    start = not start
                else:
                    pooled_scores = torch.cat((pooled_scores, pooled_score.unsqueeze(0)), dim=0)

            prediction_scores.extend(F.softmax(pooled_scores, dim=1))

    pos_scores = []
    for tensor in prediction_scores:
        pos_scores.append(tensor.tolist()[1])
    
    return pos_scores