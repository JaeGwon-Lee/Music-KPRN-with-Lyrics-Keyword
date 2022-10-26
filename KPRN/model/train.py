import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import linecache
import pickle
import constants.consts as consts
from model import KPRN
from tqdm import tqdm
from statistics import mean



# interaction 데이터 가져오기
class TrainInteractionData(Dataset):

    def __init__(self, train_path_file, in_memory=True):
        self.in_memory = in_memory
        self.file = 'data/path_data/' + train_path_file
        self.num_interactions = 0
        self.interactions = []
        
        # with pickle
        if in_memory:
            with open(self.file, "rb") as f:
                while True:
                    try:
                        obj = pickle.load(f)
                    except EOFError:
                        break
                    self.interactions.append(obj)
            self.num_interactions = len(self.interactions)
        else:
            with open(self.file, "rb") as f:
                while True:
                    try:
                        obj = pickle.load(f)
                    except EOFError:
                        break
                    self.num_interactions += 1
    
    
    # interaction 데이터 슬라이싱
    def __getitem__(self, idx):

        if self.in_memory:
            return self.interactions[idx]
        else:
            line = linecache.getline(self.file, idx+1)
            return eval(line.rstrip("\n"))
    
    
    def __len__(self):
        
        return self.num_interactions



# DataLoader에 사용할 커스텀 collate 함수 (path 형태에 맞춤)
def my_collate(batch):

    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    
    return [data, target]



# path 길이가 긴 순서로 정렬
def sort_batch(batch, indexes, lengths):

    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    indexes_tensor = indexes[perm_idx]
    
    return seq_tensor, indexes_tensor, seq_lengths



def train(model, train_path_file, batch_size, epochs, model_path, load_checkpoint,
         not_in_memory, lr, l2_reg, gamma, no_rel, rnn_type):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is", device)
    model = model.to(device)
    
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    # Learning rate : grid search (0.001, 0.002, 0.01, 0.02)
    # l2 regularization : grid search (10−5 , 10−4 , 10−3 , 10−2) -> Weight Decay (overfitting 방지)

    if load_checkpoint:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # DataLoader
    interaction_data = TrainInteractionData(train_path_file, in_memory=not not_in_memory)    # 패딩된 (path, tag, length) 형태
    train_loader = DataLoader(dataset=interaction_data, collate_fn = my_collate, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        print("Epoch is:", epoch+1)
        losses = []
        
        for interaction_batch, targets in tqdm(train_loader):
            paths = []
            lengths = []
            inter_ids = []
            for inter_id, interaction_paths in enumerate(interaction_batch):
                for path, length in interaction_paths:
                    paths.append(path)
                    lengths.append(length)
                inter_ids.extend([inter_id for i in range(len(interaction_paths))])

            # tensor로 변환
            inter_ids = torch.tensor(inter_ids, dtype = torch.long)
            paths = torch.tensor(paths, dtype=torch.long)
            lengths = torch.tensor(lengths, dtype=torch.long)

            # path 길이가 긴 순서로 정렬
            s_path_batch, s_inter_ids, s_lengths = sort_batch(paths, inter_ids, lengths)

            # gradient 초기화
            model.zero_grad()

            # Forward
            tag_scores = model(s_path_batch.to(device), s_lengths.to(device), no_rel, rnn_type)

            # 모델 점수 계산
            start = True
            for i in range(len(interaction_batch)):
                inter_idxs = (s_inter_ids == i).nonzero(as_tuple=False).squeeze(1)    # as_tuple=False 추가함

                # KPRN 유사도 점수
                pooled_score = model.weighted_pooling(tag_scores[inter_idxs], gamma=gamma)

                if start:
                    pooled_scores = pooled_score.unsqueeze(0)
                    start = not start
                else:
                    pooled_scores = torch.cat((pooled_scores, pooled_score.unsqueeze(0)), dim=0)

            prediction_scores = F.log_softmax(pooled_scores, dim=1)

            # 최적화
            loss = loss_function(prediction_scores.to(device), targets.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        print("loss is:", mean(losses))
        
        # 모델 저장
        print("Saving checkpoint to disk...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
          }, model_path)
        # torch.save(model.state_dict(), model_path)

    return model