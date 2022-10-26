import argparse
import itertools
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_input',
                       default='crawling.csv',
                       type=str)
    parser.add_argument('--file_output',
                       default='keyword.json',
                       type=str)
    return parser.parse_args()



# 가사 전처리
def preprocessing(data) :
    
    lyrics = []
    for lyric_list in list(data['lyrics']) :

        if lyric_list == 'None' :    # 가사가 없는 경우, 빈 리스트 출력
            lyrics.append(lyric_list)
            continue

        lyric_list = eval(lyric_list)    # 리스트 활성화
        
        # 가사 데이터에서 노이즈 제거
        if lyric_list[0] == ' height:auto; 로 변경시, 확장됨 ' :
            
            lyric = ''
            for l in lyric_list[1:] :
                
                if lyric == '' :
                    l = l.strip().replace('\r', '').replace('\n', '').replace('\t', '').replace('`', "'")
                else :
                    l = l.strip().replace('`', "'")
                    
                lyric = lyric + l
                
                if l != '' :
                    lyric = lyric + '\n'
                    
            lyric = lyric.rstrip('\n')    # 마지막 \n 삭제
            
        else :
            lyric = ''
            for l in lyric_list :
                l = l.strip().replace('`', "'")
                lyric = lyric + l
                
                if l != '' :
                    lyric = lyric + '\n'
                    
            lyric = lyric.rstrip('\n')
            
        lyrics.append(lyric)
        
    return lyrics



# 유사도 계산
def max_similarity(doc_embedding, candidate_embeddings, candidates, top_n):
    
    # 문서와 각 키워드들 간의 유사도
    distances = util.cos_sim(doc_embedding, candidate_embeddings)

    # 각 키워드들 간의 유사도
    distances_candidates = util.cos_sim(candidate_embeddings, candidate_embeddings)

    # 코사인 유사도에 기반하여 키워드들 중 상위 top_n개의 단어를 pick.
    words_idx = list(distances.argsort()[0][-top_n:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    return words_vals



# 키워드 추출
def get_keywords(lyrics) :
    
    model = SentenceTransformer('jhgan/ko-sroberta-multitask').to('cuda')
    okt = Okt()
    keywords = []

    for doc in tqdm(lyrics) :

        if len(doc) < 100 :    # 가사 길이가 너무 짧은 경우 제외
            doc = 'None'

        if doc == 'None' :    # 가사가 없는 경우, 빈 리스트 출력
            keywords.append([])
            continue

        # 한글은 명사만 출력
        tokenized_doc = okt.pos(doc, norm=True, stem=True)
        tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if (word[1] == 'Noun') or (word[1] == 'Alpha') or (word[0] == "'")])
        tokenized_nouns = tokenized_nouns.replace(" ' ", "'")

        # 영어는 불용어 제거
        try :
            count = CountVectorizer(stop_words='english').fit([tokenized_nouns])
            candidates = count.get_feature_names_out()
        except :
            keywords.append([])
            continue

        if len(candidates) < 5 :    # 추출된 단어가 5개 미만인 경우 제외
            keywords.append([])
            continue

        # 가사 전체와 단어 임베딩
        doc_embedding = model.encode([doc])
        candidate_embeddings = model.encode(candidates)

        # 유사도 계산
        keyword = max_similarity(doc_embedding, candidate_embeddings, candidates, top_n=5)
        keywords.append(keyword)

    return keywords



# 키워드 저장
def save_json(data, keywords, file_name) :
    
    dict_ = []
    for song_num, keyword in zip(list(data['song_num']), keywords) :
        dict_.append({'song_id': song_num, 'keyword': keyword})

    with open('../res/keyword/'+ file_name, 'w') as f:
        json.dump(dict_, f, ensure_ascii=False)



def main() :
    
    args = parse_args()
    data = pd.read_csv('../res/lyrics/' + args.file_input)
    
    lyrics = preprocessing(data)
    keywords = get_keywords(lyrics)
    
    save_json(data, keywords, args.file_output)



if __name__ == "__main__":
    main()