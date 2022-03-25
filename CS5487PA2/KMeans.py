from numpy import *

def findcent(data,n):
    dataC,ltmp = data.shape
    cent = zeros((n,ltmp))
    for i in range(n):
        k= int(random.uniform(0,dataC))
        cent[i,:]=data[k,:]
    return cent

def dist(m,n):
    return sqrt(sum(power(m-n,2)))

def K_Means(data, k):
    DataC = data.shape[0]
    Lab = mat(zeros((DataC, 2)))
    Cent = findcent(data, k)
    TmpJ = True
    while TmpJ:
        TmpJ = False
        for i in range(DataC):
            A = 0
            B = 100000.0
            for j in range(k):
                Len = dist(Cent[j, :], data[i, :])
                if Len < B:
                    A = j
                    B = Len
            if Lab[i, 0] != A:
                Lab[i, :] = A, B ** 2
                TmpJ = True
        for m in range(k):
            point = data[nonzero(Lab[:, 0].A == m)[0]]
            Cent[m, :] = mean(point, axis=0)
    return Cent, Lab