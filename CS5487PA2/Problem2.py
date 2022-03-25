import pa2
import numpy as np
import pylab as pl
from PIL import Image
import KMeans
import EMGMM
import Meanshift
import scipy.cluster.vq as vq
import sklearn.cluster as ms

def KMeansF(data,k):
    datatmp = np.mat(data)
    cent, label = KMeans.K_Means(datatmp, k)
    cluA = np.array(label)
    Klabel = cluA[:, 0]
    return Klabel

def GMMEMF(data,k,iter):
    datatmp = np.mat(data)
    mu, cov, alpha = EMGMM.GMMEM(datatmp, k, iter)
    Gm = EMGMM.Estep(datatmp, mu, cov, alpha)
    GmLabel = Gm.argmax(axis=1).flatten().tolist()[0]
    return np.array(GmLabel)

def MeanShiftF(data):
    MeanSclf = ms.MeanShift(bandwidth=1, n_jobs=-1)
    MeanSclf.fit(data)
    Label = MeanSclf.labels_
    return np.array(Label)

if __name__ == '__main__':
    PicNo='21077'
    img = Image.open('images/'+PicNo+'.jpg')
    X, L = pa2.getfeatures(img, 7)

    # KMeans
    KMeans_Y= KMeansF(vq.whiten(X.T), 6)
    Y = KMeans_Y + 1
    pl.figure(1)
    pl.subplot(1,3,1)
    pl.imshow(img)
    segm = pa2.labels2seg(Y, L)
    pl.subplot(1, 3, 2)
    pl.imshow(segm)
    csegm = pa2.colorsegms(segm, img)
    pl.subplot(1,3,3)
    pl.imshow(csegm)

    # GMMEM
    GMMEM_Y=GMMEMF(vq.whiten(X.T), 6, 200)
    Y = GMMEM_Y + 1
    pl.figure(2)
    pl.subplot(1,3,1)
    pl.imshow(img)
    segm = pa2.labels2seg(Y, L)
    pl.subplot(1, 3, 2)
    pl.imshow(segm)
    csegm = pa2.colorsegms(segm, img)
    pl.subplot(1,3,3)
    pl.imshow(csegm)

    # MeanShift
    MeanShift_Y=MeanShiftF(vq.whiten(X.T))
    Y = MeanShift_Y + 1
    pl.figure(3)
    pl.subplot(1,3,1)
    pl.imshow(img)
    segm = pa2.labels2seg(Y, L)
    pl.subplot(1, 3, 2)
    pl.imshow(segm)
    csegm = pa2.colorsegms(segm, img)
    pl.subplot(1,3,3)
    pl.imshow(csegm)

    DataTmp=vq.whiten(X.T)
    m=0.5
    DataTmp[:, 2] *= m
    DataTmp[:, 3] *= m
    KMeans_modify_Y= KMeansF(DataTmp, 3)
    Y = KMeans_modify_Y + 1
    pl.figure(4)
    pl.subplot(1,3,1)
    pl.imshow(img)
    segm = pa2.labels2seg(Y, L)
    pl.subplot(1, 3, 2)
    pl.imshow(segm)
    csegm = pa2.colorsegms(segm, img)
    pl.subplot(1,3,3)
    pl.imshow(csegm)

    DataTmp2 = vq.whiten(X.T)
    h1=1.5
    h2=0.5
    DataTmp2[:, 0] *= h1
    DataTmp2[:, 1] *= h1
    DataTmp2[:, 2] *= h2
    DataTmp2[:, 3] *= h2
    MeanShift_modify_Y = MeanShiftF(DataTmp2)
    Y = MeanShift_modify_Y + 1
    pl.figure(5)
    pl.subplot(1,3,1)
    pl.imshow(img)
    segm = pa2.labels2seg(Y, L)
    pl.subplot(1, 3, 2)
    pl.imshow(segm)
    csegm = pa2.colorsegms(segm, img)
    pl.subplot(1,3,3)
    pl.imshow(csegm)
    pl.show()