import numpy as np
import KMeans
import EMGMM
import Meanshift
import matplotlib.pyplot as plt

COLOR = ['r', 'g', 'b','m']

def loadata(filename):
    data=[]
    file = open(filename)
    for i in file.readlines():
        tmpline = i.strip().split('\t')
        tmp = []
        for j in tmpline:
            tmp.append(float(j))
        data.append(tmp)
    file.close()
    return np.array(data)

def KmeansF(data,k,name,no):
    datatmp = np.mat(data)
    cent, label = KMeans.K_Means(datatmp, k)
    CluA = np.array(label)
    Klabel = CluA[:, 0]
    plt.figure(name)
    plt.title('Kmeans_'+no)
    for i in range(len(data)):
        plt.scatter(data[i, 0], data[i, 1], color=COLOR[int(Klabel[i])])

def GMMEMF(data,k,iter,name,no):
    datatmp = np.mat(data)
    mu, cov, alpha = EMGMM.GMMEM(datatmp, k, iter)
    Gm = EMGMM.Estep(datatmp, mu, cov, alpha)
    GmLabel = Gm.argmax(axis=1).flatten().tolist()[0]
    plt.figure(name)
    plt.scatter(data[:, 0], data[:, 1], c=GmLabel, s=40, cmap='viridis')
    plt.title("GMMEM_"+no)

def MeanShiftF(data,BW,name,no):
    MeanShift = Meanshift.MeanShift()
    __, Label, Cent = MeanShift.Output(data, BW=BW)
    plt.figure(name)
    plt.title("MeanShift_"+no)
    for i in range(len(data)):
        plt.scatter(data[i, 0], data[i, 1], color=COLOR[Label[i]])

if __name__ == '__main__':
    AxData=loadata("./cluster_data_text/cluster_data_dataA_X.txt")
    BxData=loadata("./cluster_data_text/cluster_data_dataB_X.txt")
    CxData=loadata("./cluster_data_text/cluster_data_dataC_X.txt")
    AxData=AxData.transpose()
    BxData=BxData.transpose()
    CxData=CxData.transpose()
    
    KmeansF(AxData, 4, 1, 'A')
    KmeansF(BxData, 4, 2, 'B')
    KmeansF(CxData, 4, 3, 'C')

    GMMEMF(AxData, 4, 200, 4, 'A')
    GMMEMF(BxData, 4, 200, 5, 'B')
    GMMEMF(CxData, 4, 200, 6, 'C')

    MeanShiftF(AxData, 0.1, 7, 'A')
    MeanShiftF(BxData, 0.1, 8, 'B')
    MeanShiftF(CxData, 0.1, 9, 'C')

    plt.show()
