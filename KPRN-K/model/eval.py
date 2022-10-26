import math



def hit_at_k(ranked_tuples, k):

    for (score, tag) in ranked_tuples[:k]:
        if tag == 1:
            return 1
    return 0



def ndcg_at_k(ranked_tuples, k):
    '''
    DCG_k = rel_i / log(i+1)
    IDCG_k = rel_i / log(1+1)  since the first item is best rank
    NDCG_k = DCG_k / IDCG_k = log(2) / log(i+1)
    '''
    
    for i,(score, tag) in enumerate(ranked_tuples[:k]):
        if tag == 1:                                # positive = 1 / negative = 0
            return math.log(2) / math.log(i + 2)    # 인덱스는 0부터 시작하므로 i+1 대신 i+2 사용
    return 0