'''
本文件实现了svd++算法，svd++ 认为只要用户对物体进行评分，不管评分多少，
就在一定程度上反映了他对各个隐因子的喜好程度yi= (yi1, yi2, ..., yik),k为划分的类别数
接下来看下用到的数学公式，
hat(rui) = sum((Puk + sum(Yjk)/len(N(u)))*Qik) + bu + bi + u
其中N(u)指的是用户u评价过的物品集合,
Yjk外面的sum是指j的变化，j属于N(u)，也就是j是指item

cost在svd的基础上加上lambda*Y^2

puk = puk + learning_rate*(eui * Qik - lambda*Puk)
Qik = Qik + learning_rate*(eui * (Puk + sum(Yjk)/len(N(u))) - lambda*Qik)
bu = bu + learning_rate*(eui - lambda*bu)
bi = bi + learning_rate*(eui - lambda*bi)
Yjk = 

自己实现的svd++运行时间太长，加了个Y，时间变成了svd的15倍以上的时间，不知道问题在哪

'''

import random
import operator
import math
class svdpp:
    def __init__(self, iterations, K, learning_rate, lambd):
        self.iterations = iterations
        self.K = K
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.P = dict()
        self.Q = dict()
        self.bu = dict()
        self.bi = dict()
        self.Y = dict()
        self.Allmean = 0.0
        self.AllItemSet = set()


    def InitAllItemSet(self, train):
        for user, items in train.items():
            for item, rating in items.items():
                self.AllItemSet.add(item)

    def InitItemPools(self, items):
        positive_pools = set()
        for item, rating in items.items():
            positive_pools.add(item)
        negative_pools = list(self.AllItemSet - positive_pools)
        return negative_pools

    def InitModel(self,train):
        cnt = 0
        for user, items in train.items():
            self.bu[user] = 0
            self.P[user] = dict()
            for i in range(self.K):
                self.P[user][i] = random.random()
            for item, rating in items.items():
                self.Allmean += rating
                cnt += 1
                if item not in self.bi:
                    self.bi[item] = 0
                if item not in self.Q:
                    self.Q[item] = dict()
                    for j in range(self.K):
                        self.Q[item][j] = random.random()
                if item not in self.Y:
                    self.Y[item] = dict()
                    for k in range(self.K):
                        self.Y[item][k] = 0
        self.Allmean = self.Allmean / cnt

    def SelectNegativeSamples(self, items):
        ret = dict()
        for item, rating in items.items():
            ret[item] = 1
        negative_pools = self.InitItemPools(items)
        cnt = 0 
        for i in range(len(items)*3):
            item = negative_pools[random.randint(0, len(negative_pools)-1)]
            if item in ret:
                continue
            else:
                ret[item] = 0
                cnt += 1
                if cnt>=len(items):
                    break
        return ret

    #求rui
    def Predict(self, user, item, train):
        #print(user,item)
        rating = 0.0 
        items = train[user]#items就对应着
        z = [0.0 for k in range(self.K)]
        for ri, _ in items.items():
            for i in range(self.K):
                z[i] += self.Y[ri][i]
        for j in range(self.K):
            rating = rating + (self.P[user][j] + z[j] / len(items)) * self.Q[item][j]
        return rating + self.bu[user] + self.bi[item] + self.Allmean

    def TrainModel(self, train):
        self.InitAllItemSet(train)
        self.InitModel(train)
        for iteration in range(self.iterations):
            print("iteration",iteration)
            for user, items in train.items():
                z = [0.0 for k in range(self.K)]
                for ri, _ in items.items():
                    for i in range(self.K):
                        z[i] = z[i] + self.Y[ri][i]
                s = [0.0 for k in range(self.K)]
                samples = self.SelectNegativeSamples(items)
                for item, rui in samples.items():
                    eui = rui - self.Predict(user, item, train)
                    self.bu[user] = self.bu[user] + self.learning_rate*(eui - self.lambd * self.bu[user])
                    self.bi[item] = self.bi[item] + self.learning_rate*(eui - self.lambd * self.bi[item])
                    for i in range(self.K):
                        s[i] += self.Q[item][i]*eui
                        self.P[user][i] = self.P[user][i] = self.learning_rate * (eui * self.Q[item][i]
                            - self.lambd * self.P[user][i])
                        self.Q[item][i] = self.Q[item][i] = self.learning_rate * (eui * (self.P[user][i] + 
                            z[i] / math.sqrt(len(items))) - self.lambd * self.Q[item][i])
                    for j in range(self.K):
                        self.Y[item][j] += self.learning_rate*(s[j]/math.sqrt(len(items)) - self.lambd * self.Y[item][j])
        self.learning_rate = self.learning_rate * 0.9

    def Recommend(self, user, train):
        res = dict()
        interacted_items = train[user]
        for item in self.Q.keys():
            if item in interacted_items:
                continue
            else:
                rating = 0 
                for i in range(self.K):
                    rating += self.P[user][i] * self.Q[item][i]
                res[item] = rating
        return res
    def Recommendation(self, users, train):
        rank = dict()
        for user in users:
            res = self.Recommend(user, train)
            R = sorted(res.items(), key=operator.itemgetter(1), reverse=True)
            rank[user] = R
        return rank








