import numpy as np
stopcount = 1e-4
clusteriter = 1e-1

def Kernelgausin(m, n):
    Atmp = (n * np.sqrt(2 * np.pi))
    Btmp = -0.5 * ((m / n)) ** 2
    return (1 / Atmp) * np.exp(Btmp)

def Caldist(m, n):
    dist = np.array(m) - np.array(n)
    return np.linalg.norm(dist)

class MeanShift(object):
    def __init__(self, kernel=Kernelgausin):
        self.kernel = kernel

    def process1(self, sdata, data, BW):
        row,col=data.shape
        point=np.zeros(col)
        s = 0.0
        for i in data:
            len = Caldist(sdata, i)
            Gausin = self.kernel(len, BW)
            point=point+ i * Gausin
            s = s + Gausin
        point=point/s
        return point

    def Cluster(self, dt):
        index1 = []
        index2 = 0
        cent = []
        for m, n in enumerate(dt):
            if (len(index1) == 0):
                index1.append(index2)
                cent.append(n)
                index2 = index2 + 1
            else:
                for k in cent:
                    dist = Caldist(n, k)
                    if (dist < clusteriter):
                        index1.append(cent.index(k))
                if (len(index1) < m + 1):
                    index1.append(index2)
                    cent.append(n)
                    index2 = index2 + 1
        return cent, index1

    def Output(self, data, BW):
        kk = [True] * data.shape[0]
        points = np.array(data)
        while (1):
            MaxLen = 0
            for i in range(0, len(points)):
                if not kk[i]:
                    continue
                tmp = points[i].copy()
                points[i] = self.process1(points[i], data, BW)
                dist = Caldist(points[i], tmp)
                MaxLen = max(MaxLen, dist)
                kk[i] = dist > stopcount
            if (MaxLen < stopcount):
                break
        cent, lab = self.Cluster(points.tolist())
        return points, lab, cent

