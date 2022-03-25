import numpy as np
from scipy.stats import multivariate_normal

def dataprocess(data):
    sampleNo=data.shape[1]
    for i in range(sampleNo):
        Vmax = data[:, i].max()
        Vmin = data[:, i].min()
        value=Vmax - Vmin
        data[:, i] = (data[:, i] - Vmin) / value
    return data

def findpara(s, modelNo):
    row,col = s
    mu = np.random.rand(modelNo, col)
    cov = np.array([np.eye(col)] * modelNo)
    Wtmp=[1.0 / modelNo]
    alpha = np.array(Wtmp * modelNo)
    return mu, cov, alpha

def calprob(data, m, c):
    Gsian = multivariate_normal(mean=m, cov=c)
    return Gsian.pdf(data)

def Estep(data, m, c, alpha):
    modelNo = alpha.shape[0]
    sampleNo = data.shape[0]
    gm = np.mat(np.zeros((sampleNo, modelNo)))
    ratio = np.zeros((sampleNo, modelNo))
    for k in range(modelNo):
        ratio[:, k] = calprob(data, m[k], c[k])
    ratio = np.mat(ratio)
    for i in range(modelNo):
        gm[:, i] = alpha[i] * ratio[:, i]
    for j in range(sampleNo):
        gm[j, :] /= np.sum(gm[j, :])
    return gm

def Mstep(data, Gm):
    row, col = data.shape
    modelNo = Gm.shape[1]
    mu = np.zeros((modelNo, col))
    alpha = np.zeros(modelNo)
    cov = []
    for i in range(modelNo):
        KM = np.sum(Gm[:, i])
        mu[i, :] = np.sum(np.multiply(data, Gm[:, i]), axis=0) / KM
        Tmp = (data - mu[i]).transpose() * np.multiply((data - mu[i]), Gm[:, i]) / KM
        cov.append(Tmp)
        alpha[i] = KM / row
    cov = np.array(cov)
    return mu, cov, alpha

def GMMEM(data, modelNo, s):
    Newdata = dataprocess(data)
    lH=Newdata.shape
    mu, cov, alpha = findpara(lH, modelNo)
    for i in range(s):
        Gm = Estep(Newdata, mu, cov, alpha)
        mu, cov, alpha = Mstep(Newdata, Gm)
    return mu, cov, alpha

