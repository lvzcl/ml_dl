'''
本代码通过矩阵形式来实现PersonalRank，因为之前实现的PersonalRank算法需要在二分图中进行全局迭代多次，
时间复杂度太高，使用矩阵形式可以极大的提高效率

计算方式直接套用公式，其中关键的几个元素，
r:r是一个n维的向量，每一个元素代表一个节点的PR重要度。
r0:r0只在起始节点的位置上元素为1，其他为0
alpha:alpha是随机游走的概率
M:M为转移矩阵，如果节点i到节点j有单向路径的话，Mij=1/out(i),out(i)是指i的出度，如果不存在路径(是从i指向j),则Mij=0
根据之间的数据，可以的到M矩阵为：
G = {'A': {'a': 1, 'c': 1},
         'B': {'a': 1, 'b': 1, 'c': 1, 'd': 1},
         'C': {'c': 1, 'd': 1},
         'a': {'A': 1, 'B': 1},
         'b': {'B': 1},
         'c': {'A': 1, 'B': 1, 'C': 1},
         'd': {'B': 1, 'C': 1}}

M = [[0.0, 0.0, 0.0, 0.5, 0, 0.5, 0],
     [0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
     [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
     [0.33, 0.33, 0.33, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
    ]
'''

import time 
import numpy as np
import operator
from numpy.linalg import solve


if __name__ == '__main__':
    alpha = 0.8
    nodes = ["A", "B", "C", "a", "b", "c", "d"]
    M = np.matrix([[0.0, 0.0, 0.0, 0.5, 0, 0.5, 0],
                 [0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
                 [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                 [0.33, 0.33, 0.33, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                ])
    r0 = np.matrix([[0], [0], [0], [0], [1], [0], [0]])#start 节点选择'b'
    
    #第一种方法，直接使用线性求解法，求解 r * (1-alpha*M.T) = (1-alpha)*r0,当然，(1-alpha)对实际的结果没有影响，因为我们只需知道大小关系
    n = M.shape[0]
    A = np.eye(n) - alpha * M.T
    b = (1 - alpha) * r0
    begin = time.time()
    r = solve(A, b)
    #print(r)
    end = time.time()
    print("the time of using linalg slove:", end - begin)
    rank = {}
    for i in range(n):
        rank[nodes[i]] = r[i, 0]
    res = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)
    for node in res:
        print("{}:{:.2f}".format(node[0], node[1]),end=" ")
    print()

    #还可以采用CSR法，或者gmres去求解，当然，也可以直接求逆去求解

    #我们也可以直接求出从任意节点随机游走之后的结果
    
    A = np.eye(n) - alpha * M.T
    begin = time.time()
    D = A.I# A.I * np.eye(n),np.eye(n)每一行对应相对应的r0
    end = time.time()
    print("use time: ", end-begin)
    for j in range(n):
        print(nodes[j], end="\t")
        score={}
        total = 0.0
        for i in  range(n):
            score[nodes[i]] = D[i, j]
            total += D[i, j]
        res = sorted(score.items(), key=operator.itemgetter(1), reverse=True)
        for item in res:
            print("{}:{:.2f}".format(item[0],item[1]/total), end="\t")
        print()








