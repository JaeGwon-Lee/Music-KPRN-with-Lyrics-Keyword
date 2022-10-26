import pickle
import argparse
import os
import random
import time
import mmap
import torch
import pandas as pd
import numpy as np
import constants.consts as consts
from tqdm import tqdm
from statistics import mean
from collections import defaultdict
from model import KPRN, train, predict
from model.eval import hit_at_k, ndcg_at_k



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        default=False,
                        action='store_true',
                        help='whether to train the model')
    parser.add_argument('--eval',
                        default=False,
                        action='store_true',
                        help='whether to evaluate the model')
    parser.add_argument('--model',
                        type=str,
                        default='model.pt',
                        help='name to save or load model from')
    parser.add_argument('--kg_path_file',
                        type=str,
                        default='kg_path.txt',
                        help='file name to store/load train/test paths')
    parser.add_argument('--rnn_type',
                        type=str,
                        default='lstm',
                        help='Select the type of RNN (RNN/LSTM/GRU)')
    parser.add_argument('-e','--epochs',
                        type=int,
                        default=5,
                        help='number of epochs for training model')
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=256,
                        help='batch_size')
    parser.add_argument('--lr',
                        type=float,
                        default=.002,
                        help='learning rate')
    parser.add_argument('--l2_reg',
                        type=float,
                        default=.0001,
                        help='l2 regularization coefficient')
    parser.add_argument('--gamma',
                        type=float,
                        default=1,
                        help='gamma for weighted pooling')
    parser.add_argument('--load_checkpoint',
                        default=False,
                        action='store_true',
                        help='whether to load the current model state before training')
    parser.add_argument('--not_in_memory',
                        default=False,
                        action='store_true',
                        help='denotes that the path data does not fit in memory')
    parser.add_argument('--no_rel',
                        default=False,
                        action='store_true',
                        help='Run the model without relation if True')
    parser.add_argument('--np_baseline',
                        default=False,
                        action='store_true',
                        help='Run the model with the number of path baseline if True')
    parser.add_argument('--samples',
                        type=int,
                        default=-1,
                        help='number of paths to sample for each interaction (-1 means include all paths)')
    return parser.parse_args()



def seed_all(seed) :
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)



def create_directory(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass



# path 샘플링
def sample_paths(paths, samples):
    index_list = list(range(len(paths)))
    random.shuffle(index_list)
    indices = index_list[:samples]
    return [paths[i] for i in indices]



# type, relation, entity를 임베딩한 dict 파일 로드
def load_string_to_ix_dicts():
    data_path = consts.DATA_MAPPING_DIR    # data/data_mapping/

    with open(data_path + 'type_to_ix.dict', 'rb') as handle:
        type_to_ix = pickle.load(handle)
    with open(data_path + 'relation_to_ix.dict', 'rb') as handle:
        relation_to_ix = pickle.load(handle)
    with open(data_path + 'entity_to_ix.dict', 'rb') as handle:
        entity_to_ix = pickle.load(handle)

    return type_to_ix, relation_to_ix, entity_to_ix

    
    
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines



def main():
    seed_all(24)
    args = parse_args()
    model_path = "model/" + args.model    # model/model.pt
    data_ix_path = consts.DATA_IX_DIR    # data/data_ix/

    # 데이터 로드
    t_to_ix, r_to_ix, e_to_ix = load_string_to_ix_dicts()

    # KPRN 모델 로드
    model = KPRN(consts.ENTITY_EMB_DIM, consts.TYPE_EMB_DIM, consts.REL_EMB_DIM, consts.HIDDEN_DIM,
                 len(e_to_ix), len(t_to_ix), len(r_to_ix), consts.TAG_SIZE, args.no_rel, args.rnn_type)

    # TRAIN
    if args.train:
        print("Training Starting")

        # 모델 학습
        model = train(model, args.kg_path_file, args.batch_size, args.epochs, model_path,
                      args.load_checkpoint, args.not_in_memory, args.lr, args.l2_reg, args.gamma, args.no_rel, args.rnn_type)

    # TEST
    if args.eval:
        print("Evaluation Starting")
        start_time = time.time()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device is", device)

        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(device)

        hit_at_k_scores = defaultdict(list)
        ndcg_at_k_scores = defaultdict(list)
        if args.np_baseline:
            num_paths_baseline_hit_at_k = defaultdict(list)
            num_paths_baseline_ndcg_at_k = defaultdict(list)
        max_k = 15

        file_path = consts.PATH_DATA_DIR + args.kg_path_file
        with open(file_path, 'rb') as file:
            while True:
                try:
                    line = pickle.load(file)
                    test_interactions = line
                    prediction_scores = predict(model, test_interactions, args.batch_size, device, args.no_rel, args.gamma, args.rnn_type)
                    target_scores = [x[1] for x in test_interactions]    # x[1] : positive = 1 / negative = 0

                    merged = list(zip(prediction_scores, target_scores))
                    s_merged = sorted(merged, key=lambda x: x[0], reverse=True)

                    for k in range(1, max_k+1):
                        hit_at_k_scores[k].append(hit_at_k(s_merged, k))
                        ndcg_at_k_scores[k].append(ndcg_at_k(s_merged, k))

                    if args.np_baseline:
                        random.shuffle(test_interactions)
                        s_inters = sorted(test_interactions, key=lambda x: len(x[0]), reverse=True)
                        for k in range(1, max_k+1):
                            num_paths_baseline_hit_at_k[k].append(hit_at_k(s_inters, k))
                            num_paths_baseline_ndcg_at_k[k].append(ndcg_at_k(s_inters, k))
                except EOFError:
                    break

        scores = []

        for k in hit_at_k_scores.keys():
            hit_at_ks = hit_at_k_scores[k]
            ndcg_at_ks = ndcg_at_k_scores[k]
            print()
            print(["Average hit@K for k={0} is {1:.4f}".format(k, mean(hit_at_ks))])
            print(["Average ndcg@K for k={0} is {1:.4f}".format(k, mean(ndcg_at_ks))])
            scores.append([args.model, args.kg_path_file, k, mean(hit_at_ks), mean(ndcg_at_ks)])

        if args.np_baseline:
            for k in hit_at_k_scores.keys():
                print()
                print(["Num Paths Baseline hit@K for k={0} is {1:.4f}".format(k, mean(num_paths_baseline_hit_at_k[k]))])
                print(["Num Paths Baseline ndcg@K for k={0} is {1:.4f}".format(k, mean(num_paths_baseline_ndcg_at_k[k]))])
                scores.append(['np_baseline', args.kg_path_file, k, mean(num_paths_baseline_hit_at_k[k]), mean(num_paths_baseline_ndcg_at_k[k])])

        scores_cols = ['model', 'test_file', 'k', 'hit', 'ndcg']
        scores_df = pd.DataFrame(scores, columns = scores_cols)
        scores_path = 'model_scores.csv'
        try:
            model_scores = pd.read_csv(scores_path)
        except FileNotFoundError:
            model_scores = pd.DataFrame(columns=scores_cols)
        model_scores = pd.concat([model_scores, scores_df])
        model_scores.to_csv(scores_path,index=False)
        
        end_time = time.time()
        t = end_time - start_time
        print('RunTime: %dh %dm %ds' % (t//60//60, t%60, t%60%60))


        
if __name__ == "__main__":
    main()