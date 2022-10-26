DATASET_DIR = '../res/'
DATA_DICT_DIR = 'data/data_dict/'
DATA_IX_DIR = 'data/data_ix/'
DATA_MAPPING_DIR = 'data/data_mapping/'
PATH_DATA_DIR = 'data/path_data/'
KEYWORD_DIR = '../res/keyword/'

SONG_USER_DICT = 'song_user.dict'
USER_SONG_DICT = 'user_song.dict'
SONG_ARTIST_DICT = 'song_artist.dict'
ARTIST_SONG_DICT = 'artist_song.dict'
KEYWORD_SONG_DICT = 'keyword_song.dict'
SONG_KEYWORD_DICT = 'song_keyword.dict'

TYPE_TO_IX = 'type_to_ix.dict'
RELATION_TO_IX = 'relation_to_ix.dict'
ENTITY_TO_IX = 'entity_to_ix.dict'
IX_TO_TYPE = 'ix_to_type.dict'
IX_TO_RELATION = 'ix_to_relation.dict'
IX_TO_ENTITY = 'ix_to_entity.dict'

PAD_TOKEN = '#PAD_TOKEN'
SONG_TYPE = 0
USER_TYPE = 1
ARTIST_TYPE = 2
KEYWORD_TYPE = 3
PAD_TYPE = 4

SONG_USER_REL = 0
USER_SONG_REL = 1
SONG_ARTIST_REL = 2
ARTIST_SONG_REL = 3
SONG_KEYWORD_REL = 4
KEYWORD_SONG_REL = 5
UNK_REL = 6
END_REL = 7
PAD_REL = 8

ENTITY_EMB_DIM = 64    # 64 in paper
TYPE_EMB_DIM =32    # 32 in paper
REL_EMB_DIM = 32    # 32 in paper
HIDDEN_DIM = 256    # 256 in paper
TAG_SIZE = 2    # since 0 or 1

MAX_PATH_LEN = 6
NEG_SAMPLES_TRAIN = 4    # train에서 neg paths 개수
NEG_SAMPLES_TEST = 100    # test에서 neg paths 개수

LEN_3_BRANCH = 50    # 랜덤 추출할 entity 개수
LEN_5_BRANCH_TRAIN = 6
LEN_5_BRANCH_TEST= 10