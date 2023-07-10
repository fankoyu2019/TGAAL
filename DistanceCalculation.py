import numpy as np


def Distance(ps_features, ng_features, k=5):
    ps_features = np.array(ps_features)
    ng_features = np.array(ng_features)
    total_dist_list = []
    for ps_feature in ps_features:
        dist_list = []
        for ng_feature in ng_features:
            dist = F_Divergence(ps_feature, ng_feature)
            dist_list.append(dist)
        total_dist_list.append(dist_list)
    total_dist_list = np.array(total_dist_list)
    idx_list = np.argsort(total_dist_list, axis=1)[:, :k]
    return idx_list


# 欧氏距离
def EuclideanDistance(ps_feature, ng_feature):
    dist = np.linalg.norm(ps_feature - ng_feature)
    return dist


# Pearson Correlation
def PearsonCorrelation(ps_feature, ng_feature):
    x_ = ps_feature - np.mean(ps_feature)
    y_ = ng_feature - np.mean(ng_feature)
    dist = np.dot(x_, y_) / (np.linalg.norm(x_) * np.linalg.norm(y_))
    dist = 1 - dist
    return dist


def BC(p, q):
    return np.sum(np.sqrt(p * q))


# Hellinger Distance
def HellingerDistance(p, q):
    bc = BC(p, q)
    return np.sqrt(1 - bc)


# Bhattacharyya Distance
def BhattacharyyaDistance(p, q):
    bc = BC(p, q)
    return np.log(bc)


# KL散度 xlogx; reverse KL散度 −logx ; 海林格距离 (/sqrt{x} - 1) ^2 ; 卡方距离 (t-1)^2 ; α-散度 /frac {4}{1-α^2} ( 1- x^{/frac{1+α}{2}} ) (α != +-1 )
def f(t):
    return t * np.log(t)


def F_Divergence(p, q):
    p = np.array(p)+ 1e-9
    q = np.array(q)+ 1e-9
    M = (p + q) / 2
    return np.sum(q * f(p  / q ))

# 编译距离
"""
    x : string
    y: string
    return : int 
"""
def EditDistance(x, y):
    dp = np.zeros((len(x) + 1, len(y) + 1))
    for i in range(len(x) + 1):
        dp[i][0] = i
    for j in range(len(y) + 1):
        dp[0][j] = j

    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            delta = 0 if x[i - 1] == y[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return int(dp[len(x)][len(y)])
