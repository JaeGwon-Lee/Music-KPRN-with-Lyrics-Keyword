import argparse
import os
import json
import pickle
from itertools import chain
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
import re
import html
import pandas as pd
from tqdm import tqdm
import time



def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx',
                        type=int,
                        default=-1,
                        help='default is all of songs_list')
    parser.add_argument('--end_idx',
                        type=int,
                        default=-1,
                        help='default is all of songs_list')
    return parser.parse_args()



def load_json(fname):
    
    with open(fname, encoding="utf-8") as f:
        json_obj = json.load(f)

    return json_obj

    
    
def get_lyrics(song, headers) :

    url = ''
    lyrics = ''
    match = ''
    
    try :
        title_data = str(song['song']).replace("`", "'")
        artist_data = str(song['artists'][0]).replace("`", "'")
        album_data = str(song['album']).replace("`", "'")

        # html 특수문자 변환
        if '&#' in title_data :
            title_data = html.unescape(title_data)
        if '&#' in title_data :
            artist_data = html.unescape(artist_data)
        if '&#' in title_data :
            album_data = html.unescape(album_data)
        
        # 검색어
        search = title_data.replace(' ', '+') +'%2C+'+ artist_data.replace(' ', '+') +'%2C+'+ album_data.replace(' ', '+')
        url = 'https://search.daum.net/search?w=music&m=song&nil_search=btn&DA=NTB&q=' + search
    
        # html 가져오기
        html_ = requests.get(url, headers=headers).text
        soup = BeautifulSoup(html_, "html.parser")

        # 제목 일치 여부
        title_data = re.sub("\[.*\]", "", re.sub("\(.*\)", "", title_data)).lower()
        title_data = ''.join(char for char in title_data if char.isalnum())    # 문자만 남김 (특수문자, 공백 제거)
        title_daum = soup.select('#musicNColl > div.coll_cont > ul > li.fst > div.wrap_cont > div > div > a')[0].get_text()
        title_daum = re.sub("\[.*\]", "", re.sub("\(.*\)", "", title_daum)).lower()
        title_daum = ''.join(char for char in title_daum if char.isalnum())
        cond1 = title_data == title_daum
        
        # 가수 일치 여부
        artist_data = re.sub("\[.*\]", "", re.sub("\(.*\)", "", artist_data)).lower()
        artist_data = ''.join(char for char in artist_data if char.isalnum())
        artist_daum = soup.select('#musicNColl > div.coll_cont > ul > li.fst > div.wrap_cont > div > dl:nth-child(2) > dd > a:nth-child(1)')[0].get_text()
        artist_daum = re.sub("\[.*\]", "", re.sub("\(.*\)", "", artist_daum)).lower()
        artist_daum = ''.join(char for char in artist_daum if char.isalnum())
        cond2 = artist_data == artist_daum
        
        # 앨범 일치 여부
        album_data = re.sub("\[.*\]", "", re.sub("\(.*\)", "", album_data)).lower()
        album_data = ''.join(char for char in album_data if char.isalnum())
        album_daum = soup.select('#musicNColl > div.coll_cont > ul > li.fst > div.wrap_cont > div > dl:nth-child(3) > dd > a')[0].get_text()
        album_daum = re.sub("\[.*\]", "", re.sub("\(.*\)", "", album_daum)).lower()
        album_daum = ''.join(char for char in album_daum if char.isalnum())
        cond3 = album_data == album_daum
        
        # 발매일 일치 여부
        if len(song['issue_date']) != 8 :
            song['issue_date'] = song['issue_date'].ljust(8, '0')
        date_data = song['issue_date'][:4] + '.' + song['issue_date'][4].replace('0', '') + song['issue_date'][5] + '.' + song['issue_date'][6].replace('0', '') + song['issue_date'][7] + '.'    # ex) 2022.04.24.
        date_data = date_data.replace('..', '.')    # day가 없는 경우 대비
        date_daum = soup.select('#musicNColl > div.coll_cont > ul > li.fst > div.wrap_cont > div > dl:nth-child(4) > dd > span:nth-child(1)')[0].get_text()
        cond4 = date_data == date_daum

        if cond1 & cond2 & cond3 & cond4 :
            match = 'Pass'
        else :
            match = 'Miss'
        
        if soup.select('#musicNColl > div.coll_cont > ul > li.fst > div.wrap_cont > div > div > a.btn_flex3 > span') == [] :    # 검색 첫 곡에 가사가 없는 경우
            lyrics = 'None'
        else :
            lyrics = soup.find('p', {'class' : 'txt_lyrics'}).find_all(text=True)
                
    except :
        if lyrics == '' :
            lyrics = 'Error'
        if match == '' :
            match = 'Error'

    time.sleep(0.5)

    return lyrics, match, url



def save_data(list_, start_idx, end_idx) :
    
    df_ = pd.DataFrame(list_)
    
    file_name = './lyrics/daum_' + str(start_idx) + '-' + str(end_idx) + '.csv'
    df_.to_csv(file_name, encoding='utf-8', index=False)
    
    
    
def crawling(songs_list, start_idx, end_idx) :
    
    songs = load_json('./song_meta.json')
    
    # melon 봇 차단 해결
    headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'}
    
    list_ = [] ; cnt = 1
    for s in tqdm(songs_list[start_idx:end_idx]) :
        
        song = songs[s]
        
        dict_ = {}
        dict_['song_num'] = song['id']
        dict_['song'] = song['song_name']
        dict_['artists'] = song['artist_name_basket']
        dict_['album'] = song['album_name']
        dict_['genre'] = song['song_gn_gnr_basket']
        dict_['issue_date'] = song['issue_date']
        
        lyrics, match, url = get_lyrics(dict_, headers)
        
        dict_['lyrics'] = lyrics
        dict_['match'] = match
        dict_['url'] = url
    
        list_.append(dict_)
        
        # 400회마다 데이터 저장
        if cnt % 1000 == 0 :
            save_data(list_, start_idx, end_idx)
        cnt += 1
            
    save_data(list_, start_idx, end_idx)



def main() :
    
    args = parse_args()
    
    df_songs = pd.read_csv('./lyrics/crawling_list.csv')
    songs_list = list(df_songs['song_num'])
    
    # 크롤링 개수 지정
    if args.start_idx == -1 :
        start_idx = 0
    else :
        start_idx = args.start_idx
    if args.end_idx == -1 :
        end_idx = len(songs_list)
    else :
        end_idx = args.end_idx
    
    crawling(songs_list, start_idx, end_idx)

    
    
if __name__ == "__main__":
    main()