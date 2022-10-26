import pickle
import argparse
import os
import random
import copy
import constants.consts as consts
from tqdm import tqdm
from collections import defaultdict



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        default=False,
                        action='store_true')
    parser.add_argument('--eval',
                        default=False,
                        action='store_true')
    parser.add_argument('--kg_path_file',
                        type=str,
                        default='kg_path.pkl')
    parser.add_argument('--samples',
                        type=int,
                        default=-1,
                        help='number of paths to sample for each interaction (-1 means include all paths)')
    return parser.parse_args()



def seed_all(seed) :
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)



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



# relation이 저장된 dict 파일 로드
def load_rel_ix_dicts():
    data_path = consts.DATA_IX_DIR    # data/data_ix/

    with open(data_path + 'ix_song_user.dict', 'rb') as handle:
        song_user = pickle.load(handle)
    with open(data_path + 'ix_user_song.dict', 'rb') as handle:
        user_song = pickle.load(handle)
    with open(data_path + 'ix_song_artist.dict', 'rb') as handle:
        song_artist = pickle.load(handle)
    with open(data_path + 'ix_artist_song.dict', 'rb') as handle:
        artist_song = pickle.load(handle)
        
    return song_user, user_song, song_artist, artist_song



# path 형식 정의
class PathState:
    def __init__(self, path, length, entities):
        self.path = path    # [entity, type, relation to next]
        self.length = length
        self.entities = entities



# 인덱스 랜덤 추출
def get_random_index(nums, max_length):
    index_list = list(range(max_length))
    random.shuffle(index_list)
    
    return index_list[:nums]



# 한 user에 대한 path 생성
def find_paths_user_to_songs(start_user, song_artist, artist_song, 
                             song_user, user_song, max_length, sample_nums):
    
    song_to_paths = defaultdict(list)
    stack = []
    start = PathState([[start_user, consts.USER_TYPE, consts.END_REL]], 0, {start_user})    # (path, length, entities)
    stack.append(start)
    while len(stack) > 0:
        front = stack.pop()    # path 리스트에서 마지막 원소 추출
        entity, type = front.path[-1][0], front.path[-1][1]
        
        # song으로 끝나면서 지정 길이에 도달한 path 저장
        if type == consts.SONG_TYPE and front.length == max_length:
            song_to_paths[entity].append(front.path)

        # path가 song으로 끝나지 않으면 제외
        if front.length == max_length:
            continue

        # type이 user이면서 user가 선호하는 song인 경우
        if type == consts.USER_TYPE and entity in user_song:
            song_list = user_song[entity]
            index_list = get_random_index(sample_nums, len(song_list))    # user가 선호하는 song 랜덤 추출
            for index in index_list:
                song = song_list[index]
                if song not in front.entities:
                    new_path = copy.deepcopy(front.path)
                    new_path[-1][2] = consts.USER_SONG_REL
                    new_path.append([song, consts.SONG_TYPE, consts.END_REL])
                    new_state = PathState(new_path, front.length + 1, front.entities|{song})
                    stack.append(new_state)
                    
        # type이 song인 경우
        elif type == consts.SONG_TYPE:
            if entity in song_user:
                user_list = song_user[entity]
                index_list = get_random_index(sample_nums, len(user_list))
                for index in index_list:
                    user = user_list[index]
                    if user not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.SONG_USER_REL
                        new_path.append([user, consts.USER_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities|{user})
                        stack.append(new_state)
            if entity in song_artist:
                artist_list = song_artist[entity]
                index_list = get_random_index(sample_nums, len(artist_list))
                for index in index_list:
                    artist = artist_list[index]
                    if artist not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.SONG_ARTIST_REL
                        new_path.append([artist, consts.ARTIST_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities|{artist})
                        stack.append(new_state)
                        
        # type이 artist인 경우
        elif type == consts.ARTIST_TYPE and entity in artist_song:
            song_list = artist_song[entity]
            index_list = get_random_index(sample_nums, len(song_list))
            for index in index_list:
                song = song_list[index]
                if song not in front.entities:
                    new_path = copy.deepcopy(front.path)
                    new_path[-1][2] = consts.ARTIST_SONG_REL
                    new_path.append([song, consts.SONG_TYPE, consts.END_REL])
                    new_state = PathState(new_path, front.length + 1, front.entities|{song})
                    stack.append(new_state)

    return song_to_paths



# path 패딩
def pad_path(seq, e_to_ix, t_to_ix, r_to_ix, max_len, padding_token):
    relation_padding =  r_to_ix[padding_token]    # [relation, type, entity] 각 임베딩 시 사용한 padding token 값
    type_padding = t_to_ix[padding_token]
    entity_padding = e_to_ix[padding_token]

    while len(seq) < max_len:
        seq.append([entity_padding, type_padding, relation_padding])    # max len까지 padding token 추가

    return seq



# path 형식 변환
def format_paths(paths, e_to_ix, t_to_ix, r_to_ix):
    new_paths = []
    for path in paths:
        path_len = len(path)
        pad_path(path, e_to_ix, t_to_ix, r_to_ix, consts.MAX_PATH_LEN, consts.PAD_TOKEN)    # path에 padding token 추가
        new_paths.append((path, path_len))    # 새로운 path 변수에 padding된 path와 padding을 제외한 길이 입력
    return new_paths



# path 생성
def load_data(song_artist, artist_song, 
              song_user_all, user_song_all, song_user_split, user_song_split, 
              e_to_ix, t_to_ix, r_to_ix, 
              kg_path_file, neg_samples, len_3_branch, len_5_branch, 
              version="train", samples=-1):

    path_dir = consts.PATH_DATA_DIR
    create_directory(path_dir)    # path 저장할 디렉터리 생성
    path_file = open(path_dir + kg_path_file, 'wb')    # path 저장할 파일 생성 (with pickle)

    # path 통계
    pos_interactions_not_found = 0    # pos path가 없는 user 수
    total_pos_interactions = 0    # pos interaction 개수
    total_neg_interactions = 0    # neg interaction 개수
    num_pos_paths = 0    # pos path 개수
    num_neg_paths = 0    # neg path 개수
    
    # user 반복문
    for user, pos_songs in tqdm(list(user_song_split.items())) :    # relation 데이터 사용
        total_pos_interactions += len(pos_songs)
        song_to_paths, neg_songs_with_paths = None, None
        cur_index = 0
        
        # song 반복문
        for pos_song in pos_songs :
            interactions = []    # test paths 저장

            # 각 user 마다 path 생성
            if song_to_paths is None:    # positive song 첫 반복에만 실행
                if version == "train":
                    song_to_paths = find_paths_user_to_songs(user, song_artist, artist_song, 
                                                             song_user_split, user_song_split, 3, len_3_branch)    # train에서는 user_song train 사용
                    song_to_paths_len5 = find_paths_user_to_songs(user, song_artist, artist_song, 
                                                                  song_user_split, user_song_split, 5, len_5_branch)
                else:
                    song_to_paths = find_paths_user_to_songs(user, song_artist, artist_song, 
                                                             song_user_all, user_song_all, 3, len_3_branch)    # test에서는 user_song 전체를 사용
                    song_to_paths_len5 = find_paths_user_to_songs(user, song_artist, artist_song, 
                                                                  song_user_all, user_song_all, 5, len_5_branch)
                for song in song_to_paths_len5.keys():
                    song_to_paths[song].extend(song_to_paths_len5[song])    # 길이 3인 path와 길이 5인 path 통합

                # negative paths
                all_pos_songs = set(user_song_all[user])    # user의 positive song 리스트
                songs_with_paths = set(song_to_paths.keys())    # path가 있는 song 리스트
                neg_songs_with_paths = list(songs_with_paths.difference(all_pos_songs))    # path가 있지만 positive song이 아닌 리스트 : negative path
                random.shuffle(neg_songs_with_paths)

            # interactions에 pos path 입력
            pos_paths = song_to_paths[pos_song]    # 찾은 path 중 해당 positive song의 path 리스트
            if len(pos_paths) > 0:
                if samples != -1:    # -1이면 모든 path 포함, 아니면 samples개 샘플링
                    pos_paths = sample_paths(pos_paths, samples)    # path 샘플링
                interaction = (format_paths(pos_paths, e_to_ix, t_to_ix, r_to_ix), 1)    # padding 수행
                                                                                         # 형태 : ((padding된 path, padding 제외한 path 길이), 1)
                                                                                         # 1 : positive / 0 : negative
                if version == "train":
                    pickle.dump(interaction, path_file, protocol=pickle.HIGHEST_PROTOCOL)    # 해당 positive song의 path 저장 (with pickle)
                else:
                    interactions.append(interaction)    # test이면 파일에 입력하지 않고, interactions 변수에 추가
                num_pos_paths += len(pos_paths)
            else:
                pos_interactions_not_found += 1
                continue    # positive path가 없으면 negative path도 건너뜀

            # path가 있는 negative interaction 추가
            found_all_samples = True
            for i in range(neg_samples):    # neg path 개수 : train 4개 / test 100개
                
                # negative path가 부족한 경우
                if cur_index >= len(neg_songs_with_paths):
                    # print("not enough neg paths, only found:", str(i))
                    found_all_samples = False
                    break
                    
                neg_song = neg_songs_with_paths[cur_index]
                neg_paths = song_to_paths[neg_song]

                if samples != -1:    # -1이면 모든 path 포함, 아니면 samples개 샘플링
                    neg_paths = sample_paths(neg_paths, samples)
                interaction = (format_paths(neg_paths, e_to_ix, t_to_ix, r_to_ix), 0)
                if version == "train":
                    pickle.dump(interaction, path_file, protocol=pickle.HIGHEST_PROTOCOL)    # with pickle
                else:
                    interactions.append(interaction)
                num_neg_paths += len(neg_paths)
                total_neg_interactions += 1
                cur_index += 1

            if found_all_samples and version == "test":
                pickle.dump(interactions, path_file, protocol=pickle.HIGHEST_PROTOCOL)    # with pickle
    
    path_file.close()

    print("number of pos interactions attempted to find:", total_pos_interactions)
    print("number of pos interactions not found:", pos_interactions_not_found)
    print("number of pos interactions:", (total_pos_interactions - pos_interactions_not_found))
    print("number of neg interactions:", total_neg_interactions)
    print("number of pos paths:", num_pos_paths)
    print("number of neg paths:", num_neg_paths)
    print("avg num paths per positive interaction:", num_pos_paths / (total_pos_interactions - pos_interactions_not_found))
    print("avg num paths per negative interaction:", num_neg_paths / total_neg_interactions)
    
    return



def main():
    seed_all(24)
    args = parse_args()
    data_ix_path = consts.DATA_IX_DIR    # data/data_ix/

    # 데이터 로드
    t_to_ix, r_to_ix, e_to_ix = load_string_to_ix_dicts()
    song_user, user_song, song_artist, artist_song = load_rel_ix_dicts()

    if args.train:
        print("Finding paths")

        # train relation 데이터 로드
        with open(data_ix_path + 'train_ix_song_user.dict', 'rb') as handle:
            song_user_train = pickle.load(handle)
        with open(data_ix_path + 'train_ix_user_song.dict', 'rb') as handle:
            user_song_train = pickle.load(handle)

        # path 생성
        load_data(song_artist, artist_song, 
                  song_user, user_song, song_user_train, user_song_train, 
                  e_to_ix, t_to_ix, r_to_ix, 
                  args.kg_path_file, consts.NEG_SAMPLES_TRAIN, consts.LEN_3_BRANCH, consts.LEN_5_BRANCH_TRAIN, 
                  version="train", samples=args.samples)
            
    if args.eval:
            print("Finding Paths")
            
            with open(data_ix_path + 'test_ix_song_user.dict', 'rb') as handle:
                song_user_test = pickle.load(handle)
            with open(data_ix_path + 'test_ix_user_song.dict', 'rb') as handle:
                user_song_test = pickle.load(handle)

            load_data(song_artist, artist_song, 
                      song_user, user_song, song_user_test, user_song_test, 
                      e_to_ix, t_to_ix, r_to_ix, 
                      args.kg_path_file, consts.NEG_SAMPLES_TEST, consts.LEN_3_BRANCH, consts.LEN_5_BRANCH_TEST, 
                      version="test", samples=args.samples)



if __name__ == "__main__":
    main()