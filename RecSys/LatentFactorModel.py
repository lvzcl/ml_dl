'''
此文件实现了带有正则项的隐语义模型
'''

import random
import operator
class LatentFactorModel:
    def __init__(self, T, K, learning_rate, lambd):
        self.K = K#K是隐语义模型中分类的数量
        self.T = T#T是训练的step
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.AllItemSet = set()
        self.P = dict()
        self.Q = dict()

    def InitModel(self,train):
        for user,items in train.items():
            self.P[user] = dict()
            for i in range(self.K):
                self.P[user][i] = random.random()
            for item,r in items.items():
                if item not in self.Q:
                    self.Q[item] = dict()
                    for j in range(self.K):
                        self.Q[item][j]=random.random()

    def InitAllItemSet(self, train):
        for user, items in train.items():
            for item, rating in items.items():
                self.AllItemSet.add(item)

    #返回负样本库，用于随机选取负样本
    def InitItems_Pool(self, items):
        postive_items = set()
        for item in items.keys():
            postive_items.add(item)
        negative_pool = list(self.AllItemSet - postive_items)
        return negative_pool


    def RandSelectNegativeSample(self,items):
        ret = dict()
        for i in items.keys():
            ret[i] = 1
        n = 0#统计负样本的数量
        items_pool = self.InitItems_Pool(items)
        for i in range(len(items)*3):
            item = items_pool[random.randint(0, len(items_pool)-1)]
            if item in ret:
                continue
            ret[item] = 0
            n += 1
            if n > len(items):
                break
        return ret

    def Train_model(self, train_data):
        self.InitAllItemSet(train_data)
        self.InitModel(train_data)
        for step in range(self.T):
            for user, items in train_data.items():
                samples = self.RandSelectNegativeSample(items)
                for item, rui in samples.items():
                    eui = rui - self.Predict(user, item)
                    for f in range(self.K):
                        self.P[user][f] = self.P[user][f] + self.learning_rate * (eui*self.Q[item][f] - self.lambd*self.P[user][f])
                        self.Q[item][f] = self.Q[item][f] + self.learning_rate * (eui*self.P[user][f] - self.lambd*self.Q[item][f])
            self.learning_rate *= 0.9

    def Predict(self, user, item):
        rating = 0 
        for i in range(self.K):
            rating += self.P[user][i] * self.Q[item][i]
        return rating

    def Recommend(self,user,train_data):
        rank = dict()
        interacted_items = train_data[user]
        for i in self.Q:
            if i in interacted_items.keys():
                continue
            rank.setdefault(i,0)
            for f,qif in self.Q[i].items():
                puf = self.P[user][f]
                rank[i] += puf * qif
        return rank

    def Recommendation(self, users, train_data):
        result = dict()
        for user in users:
            rank = self.Recommend(user, train_data)
            R = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)
            result[user] = R
        return result





