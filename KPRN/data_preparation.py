import argparse
import os
import random
import json
import pickle
import pandas as pd
import constants.consts as consts
from tqdm import tqdm
from collections import defaultdict



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--songs_file',
                        default='song_meta.json',
                        type=str)
    parser.add_argument('--interactions_file',
                        default='train.json',
                        type=str)
    parser.add_argument('--user_limit',
                        type=int,
                        default=5000)
    return parser.parse_args()



def seed_all(seed) :
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    

    
# relation 딕셔너리 생성
def song_data_prep(songs_file, interactions_file, export_dir, user_limit):
    
    # song meta 데이터
    with open(songs_file, encoding="utf-8") as f:
        songs = json.load(f)
    songs = pd.DataFrame(songs)
    songs.rename(columns={'id':'song_id', 'artist_name_basket':'artist_list'}, inplace=True)
    
    # playlist 데이터
    with open(interactions_file, encoding="utf-8") as f:
        playlist = json.load(f)
    interactions = []
    for p in playlist[:user_limit] :
        for s in p['songs'] :
            interaction = {'user' : p['id'], 'song' : s}
            interactions.append(interaction)
    interactions = pd.DataFrame(interactions)    # columns : user, song
    

    # song_artist.dict
    song_artist = songs[['song_id', 'artist_list']]
    song_artist_dict = song_artist.set_index('song_id')['artist_list'].to_dict()    # 딕셔너리로 (key : song_id , value : artist_list)
    with open(export_dir + consts.SONG_ARTIST_DICT, 'wb') as handle:
        pickle.dump(song_artist_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)    # 딕셔너리를 이진파일로 저장

    # artist_song.dict
    artist_song_dict = {}
    for row in song_artist.iterrows():
        for artist in row[1]['artist_list']:
            if artist not in artist_song_dict:
                artist_song_dict[artist] = []
            artist_song_dict[artist].append(row[1]['song_id'])
    with open(export_dir + consts.ARTIST_SONG_DICT, 'wb') as handle:
        pickle.dump(artist_song_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # song_user.dict
    song_user = interactions[['song', 'user']].set_index('song').groupby('song')['user'].apply(list).to_dict()
    with open(export_dir + consts.SONG_USER_DICT, 'wb') as handle:
        pickle.dump(song_user, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # user_song.dict
    user_song = interactions[['user', 'song']].set_index('user').groupby('user')['song'].apply(list).to_dict()
    with open(export_dir + consts.USER_SONG_DICT, 'wb') as handle:
        pickle.dump(user_song, handle, protocol=pickle.HIGHEST_PROTOCOL)



# KG 임베딩
def convert_to_ids(entity_to_ix, rel_dict, start_type, end_type):
    new_rel = {}
    for key, values in rel_dict.items():
        if values == [] :
            continue
        key_id = entity_to_ix[(key, start_type)]    # key 인덱싱
        value_ids = []
        for val in values:
            value_ids.append(entity_to_ix[(val, end_type)])    # value 인덱싱
        new_rel[key_id] = value_ids
    return new_rel



# 데이터 임베딩
def ix_mapping(import_dir, export_dir, mapping_export_dir):
    
    # relation 데이터 로드
    with open(import_dir + consts.SONG_USER_DICT, 'rb') as handle:    # data/data_dict/song_user.dict
        song_user = pickle.load(handle)
    with open(import_dir + consts.USER_SONG_DICT, 'rb') as handle:
        user_song = pickle.load(handle)
    with open(import_dir + consts.SONG_ARTIST_DICT, 'rb') as handle:
        song_artist = pickle.load(handle)
    with open(import_dir + consts.ARTIST_SONG_DICT, 'rb') as handle:
        artist_song = pickle.load(handle)

    songs = set(song_user.keys()) | set(song_artist.keys())
    users = set(user_song.keys())
    artists = set(artist_song.keys())
    pad_token = consts.PAD_TOKEN
    
    # type 임베딩
    type_to_ix = {'user': consts.USER_TYPE, 'song': consts.SONG_TYPE, 'artist': consts.ARTIST_TYPE, 
                  pad_token: consts.PAD_TYPE}

    # relation 임베딩
    relation_to_ix = {'song_user': consts.SONG_USER_REL, 'user_song': consts.USER_SONG_REL,
                      'song_artist': consts.SONG_ARTIST_REL, 'artist_song': consts.ARTIST_SONG_REL,
                      '#UNK_RELATION': consts.UNK_REL, '#END_RELATION': consts.END_REL, pad_token: consts.PAD_REL}
    
    # entity 임베딩
    entity_to_ix = {(song, consts.SONG_TYPE): ix for ix, song in enumerate(songs)}    # { (song, type) : index }
    entity_to_ix.update({(user, consts.USER_TYPE): ix + len(songs) for ix, user in enumerate(users)})
    entity_to_ix.update({(artist, consts.ARTIST_TYPE): ix + len(songs) + len(users) for ix, artist in enumerate(artists)})
    entity_to_ix[pad_token] = len(entity_to_ix)
    
    # 데이터 인덱싱
    ix_to_type = {v: k for k, v in type_to_ix.items()}    # { index : 데이터 }
    ix_to_relation = {v: k for k, v in relation_to_ix.items()}
    ix_to_entity = {v: k for k, v in entity_to_ix.items()}

    # 임베딩, 인덱싱 저장
    with open(mapping_export_dir + consts.TYPE_TO_IX, 'wb') as handle:    # data/data_mapping/type_to_ix.dict
        pickle.dump(type_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + consts.RELATION_TO_IX, 'wb') as handle:
        pickle.dump(relation_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + consts.ENTITY_TO_IX, 'wb') as handle:
        pickle.dump(entity_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + consts.IX_TO_TYPE, 'wb') as handle:
        pickle.dump(ix_to_type, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + consts.IX_TO_RELATION, 'wb') as handle:
        pickle.dump(ix_to_relation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + consts.IX_TO_ENTITY, 'wb') as handle:
        pickle.dump(ix_to_entity, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # KG 임베딩
    song_user_ix = convert_to_ids(entity_to_ix, song_user, consts.SONG_TYPE, consts.USER_TYPE)
    user_song_ix = convert_to_ids(entity_to_ix, user_song, consts.USER_TYPE, consts.SONG_TYPE)
    song_artist_ix = convert_to_ids(entity_to_ix, song_artist, consts.SONG_TYPE, consts.ARTIST_TYPE)
    artist_song_ix = convert_to_ids(entity_to_ix, artist_song, consts.ARTIST_TYPE, consts.SONG_TYPE)

    # KG 임베딩 저장
    ix_dir = export_dir + 'ix_'
    with open(ix_dir + consts.SONG_USER_DICT, 'wb') as handle:    # data/data_ix/ix_song_user.dict
        pickle.dump(song_user_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ix_dir + consts.USER_SONG_DICT, 'wb') as handle:
        pickle.dump(user_song_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ix_dir + consts.SONG_ARTIST_DICT, 'wb') as handle:
        pickle.dump(song_artist_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ix_dir + consts.ARTIST_SONG_DICT, 'wb') as handle:
        pickle.dump(artist_song_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)



# train/test 데이터 분할
def train_test_split(dir):
    with open(dir + 'ix_' + consts.USER_SONG_DICT, 'rb') as handle:    # data/data_ix/ix_user_song.dict
        user_song = pickle.load(handle)

    # user_song / song_user
    train_user_song = {}
    test_user_song = {}
    train_song_user = defaultdict(list)
    test_song_user = defaultdict(list)

    # user마다 80%의 song은 train, 나머지는 test
    for user in user_song:
        pos_songs = user_song[user]
        random.shuffle(pos_songs)
        cut = int(len(pos_songs) * 0.8)

        # train
        train_user_song[user] = pos_songs[:cut]
        for song in pos_songs[:cut]:
            train_song_user[song].append(user)

        # test
        test_user_song[user] = pos_songs[cut:]
        for song in pos_songs[cut:]:
            test_song_user[song].append(user)

    # 저장
    with open('%strain_ix_%s' % (dir, consts.USER_SONG_DICT), 'wb') as handle:    # data_ix/train_ix_song_user.dict
        pickle.dump(train_user_song, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%stest_ix_%s' % (dir, consts.USER_SONG_DICT), 'wb') as handle:
        pickle.dump(test_user_song, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%strain_ix_%s' % (dir, consts.SONG_USER_DICT), 'wb') as handle:
        pickle.dump(train_song_user, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%stest_ix_%s' % (dir, consts.SONG_USER_DICT), 'wb') as handle:
        pickle.dump(test_song_user, handle, protocol=pickle.HIGHEST_PROTOCOL)



def create_directory(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass


        
def main():
    seed_all(24)
    args = parse_args()
    
    print("Data preparation")
    create_directory(consts.DATA_DICT_DIR)    # data/data_dict
    
    # relation 딕셔너리 생성 후 data/data_dict에 저장
    song_data_prep(consts.DATASET_DIR + args.songs_file,           # ../res/song_meta.json
                   consts.DATASET_DIR + args.interactions_file,    # ../res/train.json
                   consts.DATA_DICT_DIR,                           # data/data_dict
                   args.user_limit)

    # 임베딩
    print("Mapping ids to indices...")
    create_directory(consts.DATA_IX_DIR)    # data/data_ix
    create_directory(consts.DATA_MAPPING_DIR)    # data/data_mapping
    ix_mapping(consts.DATA_DICT_DIR, consts.DATA_IX_DIR, consts.DATA_MAPPING_DIR)

    # train/test 나누기
    print("Creating training and testing datasets...")
    train_test_split(consts.DATA_IX_DIR)

    
    
if __name__ == "__main__":
    main()