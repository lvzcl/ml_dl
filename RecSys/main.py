'''
本函数为main主函数，函数的思路：
1.读取文件，将文件转换成list。
2.划分为文件成训练集和测试集
3.将list转换成dict(user_item)
4.模型的训练和推荐
5.进行evaluation
'''
import Evaluation
import LatentFactorModel
import random
import time
import Eval
import svd
import svdpp
#读取数据
def read_data():
    fileName = "/Users/lvzcl/Desktop/RecSys/u.data"
    data = []
    f = open(fileName, 'r')
    for line in f.readlines():
        lineArr = line.strip().split()
        data.append([lineArr[0], lineArr[1], 1.0])
    return data


#数据集的划分,使用k折交叉验证
def split_data(data, cv, k, seed=22):
    train = []
    test = []
    for x in data:
        if random.randint(0,cv-1) == k:
            test.append(x)
        else:
            train.append(x)
    return train,test

#将list转换成dict(dict)的形式
def transform2dict(data):
    ret = dict()
    for user, item, rating in data:
        if user not in ret:
            ret[user] = dict()
        ret[user][item] = rating
    return ret


#主函数
if __name__ == '__main__':
    data = read_data()
    cv = 5
    recall = 0.0
    precision = 0.0
    for i in range(cv):
        time1 = time.time()
        print(i)
        train, test = split_data(data, cv, i)
        train = transform2dict(train)
        test = transform2dict(test)
        '''
        #SVD
        svd_model = svd.svd(30, 10, 0.02, 0.01)
        svd_model.TrainModel(train)
        result = svd_model.Recommendation(test.keys(), train)
        '''
        '''
        #接下来就是进行模型的训练和预测推荐
        lfm_model=LatentFactorModel.LatentFactorModel(30, 10, 0.02, 0.01)
        lfm_model.Train_model(train)
        #print(lfm_model.__dict__)
        #print(lfm.T, lfm.K,lfm.learning_rate,lfm.lambd)
        rank = lfm_model.Recommend('2', train)
        #print("rank",rank)
        result = lfm_model.Recommendation(test.keys(), train)
        '''


        #svdpp
        svdpp_model = svdpp.svdpp(30, 10, 0.02, 0.01)
        svdpp_model.TrainModel(train)
        result = svdpp_model.Recommendation(test.keys(), train)
        #接下来就是eval
        N = 10
        precision += Eval.Precision(result, test, N)
        recall += Eval.Recall(result, test, N)
        print("precision:",precision,"recall:",recall)
        time2 = time.time()
        print('min',(time2 - time1) / 60)
    precision = precision / cv
    recall = recall /cv
    print("precision:{:.4f} recall:{:.4f} ".format(precision, recall))








