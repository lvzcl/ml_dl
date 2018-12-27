
#卡方检验
def chi2(X, y):
    # X:n_smaples * n_features
    # y:n_samples * 1
    Y = np.append(1-y, y, axis=1)
    obs = np.dot(Y.T, X)
    feature_count = X.sum(axis=0).reshape(1,-1)
    class_prob = Y.mean(axis=0).reshape(1,-1)
    excepted = np.dot(class_prob.T, feature_count)
    temp = obs - excepted
    temp **= 2
    temp = temp / excepted
    chisq = temp.sum(axis=0)
    return chisq


#皮尔逊相关系数，协方差除以标准差
def pearson(X,y):
    #协方差，假设x， y都为vector
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    corr = np.sum(np.multiply(X-X_mean, y-y_mean))
    std = np.sqrt(np.sum((X-X_mean)**2)) * np.sqrt(np.sum((y-y_mean)**2))
    return corr // std
