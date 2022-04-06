from cProfile import label
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

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

    # SVM with PCA
    PCA2 = PCA(40)
    PCA2.fit(TrainX)
    PCATrainX2 = PCA2.transform(TrainX)
    PCATestX2 = PCA2.transform(TestX)
    SVMPCA = SVC(gamma='scale', kernel="rbf")
    SVMPCA.fit(PCATrainX2, TrainY)
    SVMPCAAcc = SVMPCA.score(PCATestX2,TestY)
    print("SVM(PCA 40) Accuracy:" + str(SVMPCAAcc))
    # test challenge data
    ChallengeDataX = LoadFile("./challenge/cdigits_digits_vec.txt")
    ChallengeDataY = LoadFile("./challenge/cdigits_digits_labels.txt")
    ChallengeDataY = np.ravel(ChallengeDataY)
    TestX1 = ChallengeDataX[:150, :]
    TestY1 = ChallengeDataY[:150]
    PCATestXNew = PCA2.transform(TestX1)
    SVMPCAAccChallenge = SVMPCA.score(PCATestXNew,TestY1)
    print("Challenge Data Accuracy:" + str(SVMPCAAccChallenge))

    # NN with PCA 40
    TrainX = DataX[:2000, :]
    TrainX /= 256
    TrainY = DataY[:2000]
    TestX = DataX[2000:, :]
    TestX /= 256
    TestY = DataY[2000:]
    TrainYNew = np.zeros((2000,10))
    for i in range(2000):
        TrainYNew[i][int(TrainY[i])] = 1
    TestYNew = np.zeros((2000,10))
    for i in range(2000):
        TestYNew[i][int(TestY[i])] = 1
    PCATmp = PCA(40)
    PCATmp.fit(TrainX)
    PCATrainX = PCATmp.transform(TrainX)
    PCATestX = PCATmp.transform(TestX)
    model = Sequential()
    model.add(Dense(500,input_shape=(PCATrainX.shape[1],), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation="relu"), )
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(PCATrainX, TrainYNew, batch_size=200, epochs=400, shuffle=True, verbose=0, validation_split=0.1)
    Acc = model.evaluate(PCATestX, TestYNew, batch_size=200, verbose=1)
    print("Default PCA Loss: "+str(Acc[0]))
    print('Default PCA Accuracy:', str(Acc[1]))
    # test challenge data
    DataX = LoadFile("../input/challenge/cdigits_digits_vec.txt")
    DataY = LoadFile("../input/challenge/cdigits_digits_labels.txt")
    DataY = np.ravel(DataY)
    TestX = DataX[:150, :]
    TestX /= 256
    TestY = DataY[:150]
    TestYNew = np.zeros((150,10))
    for i in range(150):
        TestYNew[i][int(TestY[i])] = 1
    PCATestXNew = PCATmp.transform(TestX)
    Acc = model.evaluate(PCATestXNew, TestYNew, batch_size=200, verbose=1)
    print("NN Loss: "+str(Acc[0]))
    print('NN Accuracy:', str(Acc[1]))




    

