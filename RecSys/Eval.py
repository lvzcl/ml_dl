#评价函数


def getRecommend(result, user, N):
    rank = []
    count = 0
    items = result[user]
    if len(items) < N:
        rank = items
    else:
        for item, rating in items:
            rank.append([item, rating])
            count += 1
            if count >= N:
                break
    return rank

def Precision(result, test, N=20):
    hit = 0
    All = 0
    users = test.keys()
    for user in test.keys():
        items = test[user]
        rank = getRecommend(result, user, N)
        for item, rating in rank:
            if item in items:
                hit += 1
        All += N
    return hit / (All * 1.0)

def Recall(result, test, N=20):
    hit = 0 
    All = 0
    users = test.keys()
    for user in test.keys():
        items = test[user]
        rank = getRecommend(result, user, N)
        for item, rating in rank:
            if item in items:
                hit += 1
        All += len(items)
    return hit / (All*1.0)

