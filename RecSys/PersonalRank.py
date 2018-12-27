'''
PersonalRank算法，图推荐(也可以称为基于邻域的推荐算法)
关于PersonalRank算法，推荐博客https://www.cnblogs.com/zhangchaoyang/articles/5470763.html#mjx-eqn-pr
Personalrank算法不管是用户还是item，都当作节点看待
'''
import time
import operator
def PersonalRank(G, alpha, user, iterations):
    rank = dict()
    rank = {x:0 for x in G.keys()} 
    rank[user]=1
    begin = time.time()
    for k in range(iterations):
        temp = dict()
        temp = {x:0 for x in G.keys()}
        for i, items in G.items():
            #如果进行游走，套公式
            for item, weight in items.items():
                temp[item] += alpha*rank[i]/(1.0 * len(items))
        temp[user] += (1-alpha)#不进行游走，还是套用公式，只不过不进行游走的话概率为1-alpha，然后从user重新以alpha概率进行游走 
        rank = temp 
        print(rank)
        print("iteration:",k,end="  ")
        for a,b in rank.items():
            print("%s:%.3f"%(a,b),end="  ")
        print("")
    end = time.time()
    print("user-time :",end-begin)
    #res = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)

    
    return rank

if __name__ == '__main__':
    alpha = 0.8
    G = {'A': {'a': 1, 'c': 1},
         'B': {'a': 1, 'b': 1, 'c': 1, 'd': 1},
         'C': {'c': 1, 'd': 1},
         'a': {'A': 1, 'B': 1},
         'b': {'B': 1},
         'c': {'A': 1, 'B': 1, 'C': 1},
         'd': {'B': 1, 'C': 1}}
    PersonalRank(G, alpha, 'A', 1)
