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
    parser.add_argument('--filename_range',
                        type=str,
                        default='0-615142')
    parser.add_argument('--start_idx',
                        type=int,
                        default=-1,
                        help='default is all of daum crawling')
    parser.add_argument('--end_idx',
                        type=int,
                        default=-1,
                        help='default is all of daum crawling')
    return parser.parse_args()



def load_json(fname):
    
    with open(fname, encoding="utf-8") as f:
        json_obj = json.load(f)

    return json_obj



def get_song_list(df_daum) :
    
    songs_list = []
    for _, song in df_daum.iterrows() :
        if song['match'] != 'Pass' :
            songs_list.append(song['song_num'])

    return songs_list
    


def get_song_id(song, headers) :

    url = ''
    song_id = ''
    song_id_sub = ''
    song_name = ''
    title = ''
    title_sub = ''

    # album html 가져오기
    url = 'https://www.melon.com/album/detail.htm?albumId=' + str(song['album_id'])
    html_ = requests.get(url, headers=headers).text
    soup = BeautifulSoup(html_, "html.parser")

    if '존재하지 않는 앨범 정보입니다.' in str(soup) :    # 정보가 삭제된 경우
        song_id = 'Missing'

    else :
        song_name = song['song']
        if "`" in song_name :
            song_name = song_name.replace("`", "'")
        if '&#' in song_name :
            song_name = html.unescape(song_name)

        try :
            # song_id 검색
            n_songs = len(soup.select('#frm > div > table > tbody > tr'))    # 곡(칸) 개수
            
            for j in range(n_songs) :
                # 제목 추출
                title = ''
                if soup.select('#frm > div > table > tbody > tr:nth-child(' + str(j+1) + ')[data-group-items]') == [] :    # song 정보가 아니면 pass
                    pass
                else :
                    try :
                        title = soup.select('#frm > div > table > tbody > tr:nth-child(' + str(j+1) + ') > td:nth-child(4) > div > div > div:nth-child(1) > span > a')[0].get_text()
                    except :
                        title = soup.select('#frm > div > table > tbody > tr:nth-child(' + str(j+1) + ') > td:nth-child(4) > div > div > div:nth-child(1) > span > span.disabled')[0].get_text()

                # song id 추출
                if song_name == title :
                    song_detail = soup.select('#frm > div > table > tbody > tr:nth-child(' + str(j+1) + ') > td:nth-child(3) > div > a')[0]['href']
                    song_id = song_detail.replace("javascript:melon.link.goSongDetail('", '').replace("');", '')
                    break

                song_name = re.sub("\[.*\]", "", re.sub("\(.*\)", "", song_name)).lower()
                song_name = ''.join(char for char in song_name if char.isalnum())
                title = re.sub("\[.*\]", "", re.sub("\(.*\)", "", title)).lower()
                title = ''.join(char for char in title if char.isalnum())
                if ((song_name in title) or (title in song_name)) and song_id_sub == '' :    # 제목이 정확히 같지 않지만 유사한 경우 sub로 저장
                    song_detail_sub = soup.select('#frm > div > table > tbody > tr:nth-child(' + str(j+1) + ') > td:nth-child(3) > div > a')[0]['href']
                    song_id_sub = song_detail_sub.replace("javascript:melon.link.goSongDetail('", '').replace("');", '')
                    title_sub = title

            if song_id == '' :
                song_id = song_id_sub    # 정확히 같은 제목이 없을때 sub 사용
                title = title_sub

            if song_id == '' :    # song_id가 추출되지 않은 경우
                song_id = 'Error'
        except :
            song_id = 'Error'

    time.sleep(2.5)
    
    return song_id, song_name, title
    
    
    
def get_lyrics(song, song_id, headers) :

    url = ''
    lyrics = ''

    if song_id == 'Missing' :
        lyrics = 'Missing'

    elif song_id == 'Error' :
        lyrics = 'Error'

    else :
        url = 'https://www.melon.com/song/detail.htm?songId='+str(song_id)

        try :
            # html 가져오기
            html = requests.get(url, headers=headers).text
            soup = BeautifulSoup(html, "html.parser")

            # 가사 저장
            if soup.select('#d_video_summary') == [] :    # 가사가 없는 경우
                if '가사 준비중' in soup.select('#lyricArea > div')[0].get_text() :
                    lyrics = 'None'
                elif '성인' in soup.select('#lyricArea > div')[0].get_text() :
                    lyrics = 'None'
                else :
                    lyrics = 'Error'
            else :
                lyrics = soup.find('div', {'class' : 'lyric'}).find_all(text=True)

        except :
            lyrics = 'Error'    # 에러가 발생한 url 저장

        time.sleep(2.5)

    return lyrics



def get_url(song, song_id) :

    album_url = 'https://www.melon.com/album/detail.htm?albumId=' + str(song['album_id'])
        
    if str.isdigit(song_id) :
        song_url = 'https://www.melon.com/song/detail.htm?songId=' + str(song_id)
    else :
        song_url = 'NaN'
    
    return album_url, song_url



def save_data(list_, file_output) :
    
    df_ = pd.DataFrame(list_)
    df_.to_csv(file_output, encoding='utf-8', index=False)
    
    
    
def crawling(songs_list, file_output) :
    
    songs = load_json('./song_meta.json')
    
    # melon 봇 차단 해결
    headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'}
        
    list_ = [] ; cnt = 1
    
    for s in tqdm(songs_list) :
        
        song = songs[s]
        
        dict_ = {}
        dict_['song_num'] = song['id']
        dict_['song'] = song['song_name']
        dict_['artists'] = song['artist_name_basket']
        dict_['album_id'] = song['album_id']
        dict_['album'] = song['album_name']
        dict_['genre'] = song['song_gn_gnr_basket']
        
        song_id, title_data, title_crawling = get_song_id(dict_, headers)
        lyrics = get_lyrics(dict_, song_id, headers)
        album_url, song_url = get_url(dict_, song_id)
        
        dict_['title_data'] = title_data
        dict_['title_crawling'] = title_crawling
        dict_['song_id'] = song_id
        dict_['lyrics'] = lyrics
        dict_['album_url'] = album_url
        dict_['song_url'] = song_url
    
        list_.append(dict_)
        
        # 400회마다 데이터 저장
        if cnt % 400 == 0 :
            save_data(list_, file_output)
        cnt += 1
            
    save_data(list_, file_output)



def main() :
    
    args = parse_args()
    
    file_input = './lyrics/daum_' + args.filename_range +'.csv'
    df_daum = pd.read_csv(file_input)
    
    # 크롤링 개수 지정
    if args.start_idx == -1 :
        start_idx = 0
    else :
        start_idx = args.start_idx
    if args.end_idx == -1 :
        end_idx = len(df_daum)
    else :
        end_idx = args.end_idx
    
    df_daum = df_daum.iloc[start_idx:end_idx]
    songs_list = get_song_list(df_daum)
    
    file_output = './lyrics/melon_' + str(start_idx) + '-' + str(end_idx) + '.csv'
    crawling(songs_list, file_output)



if __name__ == "__main__":
    main()