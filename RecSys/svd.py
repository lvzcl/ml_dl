'''
该文件实现了svd算法
hat(rui) = sum(Puk*Qik) + bu + bi + u
cost = sum(eui)^2 + 1/2*lambda*Pu^2 + 1/2*lambda*Pi^2 + 1/2*lambda*bu^2+ 1/2*lambda*bi^2
u是全部评分的平均值
bu是对于user的偏置项
bi是对于item的偏置项

准确率太低，召回率也不高，之后参考下有没有svd调参的文章， 或者尝试gridSearch


puk = puk + learning_rate*(eui * Qik - lambda*Puk)
Qik = Qik + learning_rate*(eui * Puk - lambda*Qik)
bu = bu + learning_rate*(eui - lambda*bu)
bi = bi + learning_rate*(eui - lambda*bi)
'''
import random
import operator

class svd:
    def __init__(self, iterations, K, learning_rate, lambd):
        self.iterations = iterations#最大迭代次数
        self.K = K#K为设定的类别数
        self.learning_rate = learning_rate
        self.lambd = lambd # 偏执系数
        self.Allmean = 0.0
        self.train = dict()
        self.AllItemSet = set()
        self.P = dict()
        self.Q = dict()
        self.bu = dict()
        self.bi = dict()

    #构造全部item的集合
    def InitAllItemSet(self, train):
        for user, items in train.items():
            for item,rating in items.items():
                self.AllItemSet.add(item)

    #给定items,求出负采样的样本池
    def InitItemPools(self,items):
        positive_items = set()
        for item in items.keys():
            positive_items.add(item)
        negative_pools = list(self.AllItemSet - positive_items)
        return negative_pools

    #模型初始化
    def InitModel(self, train):
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
        self.Allmean /= cnt

    def SelectNegativeSamples(self,items):
        ret = dict()
        for i in items.keys():
            ret[i] = 1
        negative_pools = self.InitItemPools(items)
        cnt = 0
        for i in range(len(items)*3):
            item = negative_pools[random.randint(0, len(negative_pools) - 1)]
            if item in ret:
                continue
            else:
                ret[item] = 0
                cnt += 1
                if cnt >= len(items):
                    break
        return ret

    def Predict(self, user, item):
        rating = 0.0
        for i in range(self.K):
            rating += self.P[user][i] * self.Q[item][i]
        rating = rating + self.bu[user] + self.bi[item] + self.Allmean
        return rating

    def TrainModel(self, train):
        self.InitAllItemSet(train)
        self.InitModel(train)
        #print(len(self.AllItemSet),len(self.Q))
        for iteration in range(self.iterations):
            for user, items in train.items():
                samples = self.SelectNegativeSamples(items)
                for item, result in samples.items():
                    rui = self.Predict(user, item)
                    eui = result - rui
                    self.bu[user] += self.learning_rate*(eui - self.lambd)
                    self.bi[item] += self.learning_rate*(eui - self.lambd)
                    for i in range(self.K):
                        self.P[user][i] = self.P[user][i] + self.learning_rate*(eui * self.Q[item][i]
                            - self.lambd * self.P[user][i])
                        self.Q[item][i] = self.Q[item][i] + self.learning_rate*(eui * self.P[user][i]
                            - self.lambd * self.Q[item][i])
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










