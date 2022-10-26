import argparse
import pandas as pd



def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--daum_input',
                        default='daum_crawling.csv',
                        type=str)
    parser.add_argument('--melon_input',
                        default='melon_crawling.csv',
                        type=str)
    return parser.parse_args()



# Daum / Melon 크롤링 파일 병합
def concat_df(df_daum, df_melon) :
    
    df_daum = df_daum[['song_num','song','artists', 'album', 'genre', 'lyrics']].set_index('song_num')
    df_melon = df_melon[['song_num','song','artists', 'album', 'genre', 'lyrics']].set_index('song_num')
    
    for i in df_melon.index :
        df_daum.loc[i, 'lyrics'] = df_melon.loc[i, 'lyrics']
    
    df_daum = df_daum.reset_index()
    
    return df_daum



# 장르 제외
def genre_remove(df_crawling) :
    
    # 삭제할 장르 : [클래식, 재즈, 뉴에이지, J-POP, 월드뮤직, CCM, 어린이/태교, 종교음악, 국악, 뮤직테라피, 뮤지컬]
    genre_remove = ['GN1600', 'GN1700', 'GN1800', 'GN1900', 'GN2000', 'GN2100', 'GN2200', 'GN2300', 'GN2400', 'GN2800', 'GN2900']
    
    list_ = []
    for _, song in df_crawling.iterrows() :
        if (set(song['genre']) & set(genre_remove)) != set() :    # 삭제할 장르와 겹치면 T
            song['genre_remove'] = 'T'
        else :
            song['genre_remove'] = 'F'
        list_.append(song)
    data = pd.DataFrame(list_)
    
    # None:가사 없음 / Missing:곡 사라짐 / Error:크롤링 에러
    data.loc[data['lyrics'] == 'Missing', 'lyrics'] = 'None'    # 곡 정보가 없는 경우
    data.loc[data['lyrics'] == 'Error', 'lyrics'] = 'None'    # 크롤링 에러
    data.loc[data['genre_remove'] == 'T', 'lyrics'] = 'Genre'    # 장르 제외(통계)
    
    print('장르 제외 곡 수 : %d (%.1f%%)' % (sum(data['lyrics']=='Genre'), sum(data['lyrics']=='Genre') / len(data)*100))
    print('가사가 없는 곡 수 : %d (%.1f%%)' % (sum(data['lyrics']=='None'), sum(data['lyrics']=='None') / len(data)*100))
    
    data.loc[data['lyrics'] == 'Genre', 'lyrics'] = 'None'    # 장르 제외도 None으로
    
    print('전체 곡 수 : %d' % len(data))
    print('실행 곡 수 : %d (%.1f%%)' % (sum(data['lyrics']!='None'), sum(data['lyrics']!='None') / len(data)*100))
    
    return data



def main() :
    
    args = parse_args()
    
    df_daum = pd.read_csv('./lyrics/' + args.daum_input)
    df_melon = pd.read_csv('./lyrics/' + args.melon_input)
    
    df_crawling = concat_df(df_daum, df_melon)
    df_crawling = genre_remove(df_crawling)
    
    df_crawling.to_csv('./lyrics/crawling.csv', index=False, encoding='utf-8')



if __name__ == "__main__":
    main()