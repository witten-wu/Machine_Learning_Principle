from cProfile import label
import numpy as np
from sklearn.preprocessing import scale
from sklearn.linear_model import Perceptron
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def LoadFile(name):
    file = open(name,"r")
    Matrix = []
    for line in file:
        l = line.split()
        Tmplist = [float(v) for v in l]
        Matrix.append(Tmplist)
    Matrix = np.array(Matrix)
    file.close()
    return Matrix

if __name__ == "__main__":
    DataX = LoadFile("./digits4000_txt/digits4000_digits_vec.txt")
    DataY = LoadFile("./digits4000_txt/digits4000_digits_labels.txt")
    DataY = np.ravel(DataY)
    TrainX = DataX[:2000, :]
    TrainY = DataY[:2000]
    TestX = DataX[2000:, :]
    TestY = DataY[2000:]
    Dim = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    AccList = []
    Neighbour = 1

    # Step 1 Trying
    # KNN k=1
    KNN = neighbors.KNeighborsClassifier(Neighbour)
    KNN.fit(TrainX, TrainY)
    KNNAcc = KNN.score(TestX,TestY)
    print("KNN Accuracy:" + str(KNNAcc))
    # KNN add PCA:
    PCA1 = PCA(100)
    PCA1.fit(TrainX)
    PCATrainX = PCA1.transform(TrainX)
    PCATestX = PCA1.transform(TestX)
    PCAKNN = neighbors.KNeighborsClassifier(Neighbour)
    PCAKNN.fit(PCATrainX, TrainY)
    PCAAcc = PCAKNN.score(PCATestX,TestY)
    print("KNN(PCA 100) Accuracy:" + str(PCAAcc))
    
    # SVM with poly kernel
    SVMPoly = SVC(gamma='scale', kernel="poly")
    SVMPoly.fit(TrainX,TrainY)
    SVMPolyAcc = SVMPoly.score(TestX,TestY)
    print("SVM(poly kernel) Accuracy:" + str(SVMPolyAcc))
    # SVM with rbf kernel
    SVMRbf = SVC(gamma='scale', kernel="rbf")
    SVMRbf.fit(TrainX,TrainY)
    SVMRbfAcc = SVMRbf.score(TestX,TestY)
    print("SVM(rbf kernel) Accuracy:" + str(SVMRbfAcc))
    # SVM with scale (rbf kernel)
    TrainXScale = scale(TrainX)
    TestXScale = scale(TestX)
    SVMScale = SVC(gamma='scale', kernel="rbf")
    SVMScale.fit(TrainXScale,TrainY)
    SVMScaleAcc = SVMScale.score(TestXScale,TestY)
    print("SVM(scale) Accuracy:" + str(SVMScaleAcc))
    # SVM with PCA (rbf kernel)
    PCA2 = PCA(100)
    PCA2.fit(TrainX)
    PCATrainX2 = PCA2.transform(TrainX)
    PCATestX2 = PCA2.transform(TestX)
    SVMPCA = SVC(gamma='scale', kernel="rbf")
    SVMPCA.fit(PCATrainX2, TrainY)
    SVMPCAAcc = SVMPCA.score(PCATestX2,TestY)
    print("SVM(PCA 100) Accuracy:" + str(SVMPCAAcc))
    
    # Logistic Regression
    LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=10000).fit(TrainX,TrainY)
    LRAcc = LR.score(TestX,TestY)
    print("Logistic Regression Accuracy:" + str(LRAcc))
    # Logistic Regression with PCA
    PCA3 = PCA(100)
    PCA3.fit(TrainX)
    PCATrainX3 = PCA3.transform(TrainX)
    PCATestX3 = PCA3.transform(TestX)
    LRPCA = LogisticRegression(random_state=0, solver='sag', multi_class='multinomial', max_iter=10000).fit(PCATrainX3,TrainY)
    LRPCAAcc = LRPCA.score(PCATestX3,TestY)
    print("Logistic Regression(PCA 100) Accuracy:" + str(LRPCAAcc))
    
    # Perceptron
    Perct = Perceptron(tol=1e-3, random_state=0)
    Perct.fit(TrainX,TrainY)
    PerctAcc = Perct.score(TestX,TestY)
    print("Perceptron Accuracy:" + str(PerctAcc))
    # Perceptron with PCA
    PCA4 = PCA(100)
    PCA4.fit(TrainX)
    PCATrainX4 = PCA4.transform(TrainX)
    PCATestX4 = PCA4.transform(TestX)
    PerctPCA = Perceptron(tol=1e-3, random_state=0)
    PerctPCA.fit(PCATrainX4,TrainY)
    PerctPCAAcc = PerctPCA.score(PCATestX4,TestY)
    print("Perceptron(PCA 100) Accuracy:" + str(PerctPCAAcc))

    # Step 2 Choose the best parameter for PCA
    # KNN with PCA
    plt.figure(1)
    plt.title("KNN with PCA")
    for i in Dim:
        pcatmp = PCA(i)
        pcatmp.fit(TrainX)
        TrainTmp = pcatmp.transform(TrainX)
        TestTmp = pcatmp.transform(TestX)
        model = neighbors.KNeighborsClassifier(Neighbour)
        model.fit(TrainTmp, TrainY)
        Acc = model.score(TestTmp,TestY)
        AccList.append(Acc)    
    plt.plot(Dim, AccList, color='green',label='default')
    # KNN with PCA(sigmoid kernel)
    AccList = []
    for i in Dim:
        pcatmp = KernelPCA(n_components=i,kernel="sigmoid")
        TrainXScale = scale(TrainX)
        TestXScale = scale(TestX)
        pcatmp.fit(TrainXScale)
        TrainTmp = pcatmp.transform(TrainXScale)
        TestTmp = pcatmp.transform(TestXScale)
        model = neighbors.KNeighborsClassifier(Neighbour)
        model.fit(TrainTmp, TrainY)
        Acc = model.score(TestTmp,TestY)
        AccList.append(Acc)    
    plt.plot(Dim, AccList, color='blue',label='sigmoed kernel')
    # KNN with PCA(rbf kernel)
    AccList = []
    for i in Dim:
        pcatmp = KernelPCA(n_components=i,kernel="rbf")
        TrainXScale = scale(TrainX)
        TestXScale = scale(TestX)
        pcatmp.fit(TrainXScale)
        TrainTmp = pcatmp.transform(TrainXScale)
        TestTmp = pcatmp.transform(TestXScale)
        model = neighbors.KNeighborsClassifier(Neighbour)
        model.fit(TrainTmp, TrainY)
        Acc = model.score(TestTmp,TestY)
        AccList.append(Acc)    
    plt.plot(Dim, AccList, color='red',label='rbf kernel')
    plt.xlabel("Dimension")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    # SVM with PCA
    plt.figure(2)
    plt.title("SVM with PCA")
    AccList = []
    for i in Dim:
        pcatmp = PCA(i)
        pcatmp.fit(TrainX)
        TrainTmp = pcatmp.transform(TrainX)
        TestTmp = pcatmp.transform(TestX)
        model = SVC(gamma='scale', kernel="rbf")
        model.fit(TrainTmp, TrainY)
        Acc = model.score(TestTmp,TestY)
        AccList.append(Acc)    
    plt.plot(Dim, AccList, color='green',label='default')
    # SVM with PCA(sigmoid kernel)
    AccList = []
    for i in Dim:
        pcatmp = KernelPCA(n_components=i,kernel="sigmoid")
        TrainXScale = scale(TrainX)
        TestXScale = scale(TestX)
        pcatmp.fit(TrainXScale)
        TrainTmp = pcatmp.transform(TrainXScale)
        TestTmp = pcatmp.transform(TestXScale)
        model = SVC(gamma='scale', kernel="rbf")
        model.fit(TrainTmp, TrainY)
        Acc = model.score(TestTmp,TestY)
        AccList.append(Acc)    
    plt.plot(Dim, AccList, color='blue',label='sigmoed kernel')
    # SVM with PCA(rbf kernel)
    AccList = []
    for i in Dim:
        pcatmp = KernelPCA(n_components=i,kernel="rbf")
        TrainXScale = scale(TrainX)
        TestXScale = scale(TestX)
        pcatmp.fit(TrainXScale)
        TrainTmp = pcatmp.transform(TrainXScale)
        TestTmp = pcatmp.transform(TestXScale)
        model = SVC(gamma='scale', kernel="rbf")
        model.fit(TrainTmp, TrainY)
        Acc = model.score(TestTmp,TestY)
        AccList.append(Acc)    
    plt.plot(Dim, AccList, color='red',label='rbf kernel')
    plt.xlabel("Dimension")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    # Logistic Regression with PCA
    plt.figure(3)
    plt.title("Logistic Regression with PCA")
    AccList = []
    for i in Dim:
        pcatmp = PCA(i)
        pcatmp.fit(TrainX)
        TrainTmp = pcatmp.transform(TrainX)
        TestTmp = pcatmp.transform(TestX)
        model = LogisticRegression(random_state=0, solver='sag', multi_class='multinomial', max_iter=10000)
        model.fit(TrainTmp, TrainY)
        Acc = model.score(TestTmp,TestY)
        AccList.append(Acc)    
    plt.plot(Dim, AccList, color='green',label='default')
    # Logistic Regression with PCA(sigmoid kernel)
    AccList = []
    for i in Dim:
        pcatmp = KernelPCA(n_components=i,kernel="sigmoid")
        TrainXScale = scale(TrainX)
        TestXScale = scale(TestX)
        pcatmp.fit(TrainXScale)
        TrainTmp = pcatmp.transform(TrainXScale)
        TestTmp = pcatmp.transform(TestXScale)
        model = LogisticRegression(random_state=0, solver='sag', multi_class='multinomial', max_iter=10000)
        model.fit(TrainTmp, TrainY)
        Acc = model.score(TestTmp,TestY)
        AccList.append(Acc)    
    plt.plot(Dim, AccList, color='blue',label='sigmoed kernel')
    # Logistic Regression with PCA(rbf kernel)
    AccList = []
    for i in Dim:
        pcatmp = KernelPCA(n_components=i,kernel="rbf")
        TrainXScale = scale(TrainX)
        TestXScale = scale(TestX)
        pcatmp.fit(TrainXScale)
        TrainTmp = pcatmp.transform(TrainXScale)
        TestTmp = pcatmp.transform(TestXScale)
        model = LogisticRegression(random_state=0, solver='sag', multi_class='multinomial', max_iter=10000)
        model.fit(TrainTmp, TrainY)
        Acc = model.score(TestTmp,TestY)
        AccList.append(Acc)    
    plt.plot(Dim, AccList, color='red',label='rbf kernel')
    plt.xlabel("Dimension")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    # Perceptron with PCA
    plt.figure(4)
    plt.title("Perceptron with PCA")
    AccList = []
    for i in Dim:
        pcatmp = PCA(i)
        pcatmp.fit(TrainX)
        TrainTmp = pcatmp.transform(TrainX)
        TestTmp = pcatmp.transform(TestX)
        model = Perceptron(tol=1e-3, random_state=0)
        model.fit(TrainTmp, TrainY)
        Acc = model.score(TestTmp,TestY)
        AccList.append(Acc)    
    plt.plot(Dim, AccList, color='green',label='default')
    # Perceptron with PCA(sigmoid kernel)
    AccList = []
    for i in Dim:
        pcatmp = KernelPCA(n_components=i,kernel="sigmoid")
        TrainXScale = scale(TrainX)
        TestXScale = scale(TestX)
        pcatmp.fit(TrainXScale)
        TrainTmp = pcatmp.transform(TrainXScale)
        TestTmp = pcatmp.transform(TestXScale)
        model = Perceptron(tol=1e-3, random_state=0)
        model.fit(TrainTmp, TrainY)
        Acc = model.score(TestTmp,TestY)
        AccList.append(Acc)    
    plt.plot(Dim, AccList, color='blue',label='sigmoed kernel')
    # Perceptron with PCA(rbf kernel)
    AccList = []
    for i in Dim:
        pcatmp = KernelPCA(n_components=i,kernel="rbf")
        TrainXScale = scale(TrainX)
        TestXScale = scale(TestX)
        pcatmp.fit(TrainXScale)
        TrainTmp = pcatmp.transform(TrainXScale)
        TestTmp = pcatmp.transform(TestXScale)
        model = Perceptron(tol=1e-3, random_state=0)
        model.fit(TrainTmp, TrainY)
        Acc = model.score(TestTmp,TestY)
        AccList.append(Acc)    
    plt.plot(Dim, AccList, color='red',label='rbf kernel')
    plt.xlabel("Dimension")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()
