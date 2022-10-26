import argparse
import json
import pandas as pd



def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactions_file',
                        default='train.json',
                        type=str)
    return parser.parse_args()



def load_json(fname):
    
    with open(fname, encoding="utf-8") as f:
        json_obj = json.load(f)

    return json_obj



# 크롤링할 노래 번호 저장
def get_song_list(interactions_file) :
    
    # 플레이리스트 데이터 로드
    with open(interactions_file, encoding="utf-8") as f:
        playlist = json.load(f)

    # 플레이리스트에 속한 노래 저장
    df_songs = []
    for p in playlist :
        for s in p['songs'] :
            df_songs.append({'song_num' : s})
    df_songs = pd.DataFrame(df_songs)
    df_songs = df_songs.drop_duplicates(ignore_index=True)    # 여러번 포함된 노래 제거
    
    df_songs.to_csv('./lyrics/crawling_list.csv', index=False, encoding='utf-8')
    print('number of songs in playlists :', len(df_songs))
    


def main() :
    
    args = parse_args()
    get_song_list(args.interactions_file)

    
    
if __name__ == "__main__":
    main()