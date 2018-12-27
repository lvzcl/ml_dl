#虽然是准备实现ftrl算法，但是自己并没有完全理解这个算法，只是按照伪代码然后参照别人的实现自己实现一个
import numpy as np

class LR(object):

    @staticmethod
    def fn(w,x):
        return 1.0 / (1.0 + np.exp(-np.dot(w, x)))

    @staticmethod
    def loss(y, y_hat):
        return np.sum(np.nan_to_num(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)))

    @staticmethod
    def grad(y, y_hat, x):
        return (y_hat - y) * x


class FTRL(object):

    def __init__(self, dim, l1, l2, alpha, beta, decisionFuc=LR):
        self.dim = dim
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        self.beta = beta
        self.n = np.zeros(dim)
        self.z = np.zeros(dim)
        self.w = np.zeros(dim)
        self.decisionFuc = decisionFuc

    def predict(self, x):
        return self.decisionFuc.fn(self.w, x)

    def update(self, x, y):
        self.w = np.array([0 if np.abs(self.z[i]) <= self.l1 else  ((np.sign(self.z[i])*self.l1 - self.z[i])) / (
            (self.beta + np.sqrt(self.n[i])) / self.alpha + self.l2) for i in range(self.dim)])
        y_hat = self.decisionFuc.fn(self.w, x)
        g = self.decisionFuc.grad(y, y_hat, x)
        sigma = (np.sqrt(self.n + g * g) - np.sqrt(self.n)) / self.alpha
        self.z = self.z + g - sigma*self.w
        self.n = self.n + g*g
        return self.decisionFuc.loss(y, y_hat)


    def train(self, trainSet, iterations=1000000, eta=0.01, epoches=100):
        itr = 0
        n = 0
        while True:
            for x, y in trainSet:
                loss = self.update(x, y)
                if loss < eta:
                    itr += 1
                else:
                    itr = 0
                if itr >= epoches:
                    print('loss have less than', eta, "continuously for ", itr, "iterations")
                    return 
                n += 1
                if n >= iterations:
                    print('iteration ending')
                    return

class Corpus(object):
    def __init__(self, file, d):
        self.d = d
        self.file = file

    def __iter__(self):
        with open(self.file, 'r') as f_in:
            for line in f_in:
                arr = line.strip().split()
                if len(arr) >= (self.d + 1):
                    yield (np.array([float(x) for x in arr[:self.d]]), float(arr[self.d]))


if __name__ == '__main__':
    d = 4
    corpus = Corpus('./data/train.txt', d)
    ftrl = FTRL(dim = d, l1=1.0, l2=1.0, alpha=0.1, beta=1.0)
    ftrl.train(corpus, iterations=100000, eta=0.01, epoches=100)
    w = ftrl.w
    print(w)
    correct = 0
    wrong = 0
    for x, y in corpus:
        y_hat = 1.0 if ftrl.predict(x) > 0.5 else 0
        if y == y_hat:
            correct += 1
        else:
            wrong += 1
    print('correct ratio', 1.0 * correct / (correct + wrong))







