import numpy as np
import random
from sklearn import metrics as ms


class Perceptron:
    def __init__(self, maxFeature, maxLabel, idPredicate, labelDict=None):
        self.maxFeature = maxFeature
        self.maxLabel = maxLabel
        self.uniWeights = np.zeros([maxFeature, maxLabel])
        self.biWeights = np.zeros([maxLabel + 1, maxLabel + 1])  # at 0 is label START and END regarding.
        if labelDict is not None:
            self.labelDict = labelDict
        else:
            self.labelDict = [str(i) for i in range(maxLabel + 1)]
        self.idPredicate = idPredicate

    def train(self, feas, labs, posPredicates, maxIter,
              shuffle=True, alphaFunc=None, confMatFile=None, average=False):

        nSamples = len(feas)
        sampleOrder = list(range(nSamples))

        # calculate the total labels for reporting accuracy
        nLabels = sum([len(labSeq) for labSeq in labs])

        # None for constant alpha
        if alphaFunc is None:
            alphaFunc = lambda x: 1

        fullIter = True

        if average:
            trainedUniWeights = self.uniWeights
            trainedBiWeights = self.biWeights

        for ite in range(maxIter):
            if shuffle:
                random.shuffle(sampleOrder)

            # alpha of this epoch
            alpha = alphaFunc(ite)

            # accuracy counter
            nCorrect = 0

            for ns in sampleOrder:
                predSampleLabs = self._viterbi(feas[ns], posPredicates[ns])
                nCorrect += self._update(feas[ns], labs[ns], predSampleLabs, alpha)

            # epoch info
            print("Iter %4d: Accuracy=%.4f - %8d/%8d"
                  % (ite + 1, nCorrect / nLabels, nCorrect, nLabels))

            # to calculate the sum of weights for average perceptron
            if average:
                trainedUniWeights += self.uniWeights
                trainedBiWeights += self.biWeights

            # convergence detection: all labels correct
            if nCorrect == nLabels:
                print("Algorithm converges at iteration %d. All labels correctly classified." % (ite + 1))
                fullIter = False
                break

        if average:
            self.uniWeights = trainedUniWeights / (maxIter * nSamples)
            self.biWeights = trainedBiWeights / (maxIter * nSamples)
        if fullIter:
            print("Algorithm ends after %d iterations. Evaluating results on training set:" % maxIter)

        self.eval(feas, labs, posPredicates, confMatFile=confMatFile)

    def eval(self, feas, labs, posPredicates, confMatFile=None):
        predAllLabs = []
        goldAllLabs = []
        for ns in range(len(labs)):
            goldAllLabs += labs[ns]
            predAllLabs += self._viterbi(feas[ns], posPredicates[ns])
        self._showResult(goldAllLabs, predAllLabs, confMatFile)

    def predict(self, feas, posPredicates, names=True):
        predLabs = []
        for ns in range(len(feas)):
            predLabs += [self._viterbi(feas[ns], posPredicates[ns])]
        if names:
            predLabNames = [[self.labelDict[t] for t in s] for s in predLabs]
            return predLabNames
        return predLabs

    def _viterbi(self, sampleFeas, posPredicate):
        lenSeq = len(sampleFeas)
        valSheet = np.zeros([lenSeq, self.maxLabel])
        prevSheet = np.zeros([lenSeq - 1, self.maxLabel])
        for nl in range(lenSeq):
            currFea = sampleFeas[nl]
            # if at the beginning
            if nl == 0:
                for lab in range(self.maxLabel):
                    uniValue = sum(self.uniWeights[currFea, lab])
                    biValue = self.biWeights[0, lab + 1]
                    valSheet[0, lab] = uniValue + biValue
            else:
                for lab in range(self.maxLabel):
                    uniValue = sum(self.uniWeights[currFea, lab])
                    biValue = -np.inf
                    prevMax = 0
                    for prevLab in range(self.maxLabel):
                        tmp = self.biWeights[prevLab + 1, lab + 1] + valSheet[nl - 1, prevLab]
                        if tmp > biValue:
                            biValue = tmp
                            prevMax = prevLab + 1
                    valSheet[nl, lab] = uniValue + biValue
                    prevSheet[nl - 1, lab] = prevMax
            # if the current word is the predicate
            if nl == posPredicate:
                for lab in range(self.maxLabel):
                    if not lab == self.idPredicate - 1:
                        valSheet[nl, lab] = -np.inf
        # last label
        valueFinal = -np.inf
        prevMaxFinal = 0
        for prevLab in range(self.maxLabel):
            tmp = self.biWeights[prevLab + 1, 0] + valSheet[lenSeq - 1, prevLab]
            if tmp > valueFinal:
                valueFinal = tmp
                prevMaxFinal = prevLab + 1

        # sequence retrieving
        ptr = prevMaxFinal
        predLabs = [ptr]
        for nl in range(lenSeq - 2, -1, -1):
            ptr = int(prevSheet[nl, ptr - 1])
            predLabs += [ptr]
        predLabs.reverse()
        return predLabs

    def _update(self, sampleFeas, goldSampleLabs, predSampleLabs, alpha):
        nCorrect = 0
        lenSeq = len(goldSampleLabs)
        for nl in range(lenSeq):
            currFea = sampleFeas[nl]
            goldLab = goldSampleLabs[nl]
            predLab = predSampleLabs[nl]
            if goldLab == predLab:
                nCorrect += 1
                continue
            else:
                self.uniWeights[currFea, goldLab - 1] += alpha
                self.uniWeights[currFea, predLab - 1] -= alpha
                if nl == 0:
                    self.biWeights[0, goldLab] += alpha
                    self.biWeights[0, predLab] -= alpha
                else:
                    self.biWeights[goldSampleLabs[nl - 1], goldLab] += alpha
                    self.biWeights[predSampleLabs[nl - 1], predLab] -= alpha
                    if nl == lenSeq - 1:
                        self.biWeights[goldLab, 0] += alpha
                        self.biWeights[predLab, 0] -= alpha
        return nCorrect

    def _showResult(self, goldAllLabs, predAllLabs, confMatFile=None):
        print("Accuracy=%.4f" % ms.accuracy_score(goldAllLabs, predAllLabs))
        rst = ms.precision_recall_fscore_support(goldAllLabs, predAllLabs)
        print("\tP\tR\tF1")
        for i in range(self.maxLabel):
            print("%s\t%.4f\t%.4f\t%.4f"
                  % (self.labelDict[i + 1], rst[0][i], rst[1][i], rst[2][i]))
        macro = ms.precision_recall_fscore_support(goldAllLabs, predAllLabs, average='macro')
        print("%s\t%.4f\t%.4f\t%.4f"
              % ('macro', macro[0], macro[1], macro[2]))
        if confMatFile is not None:
            np.savetxt(confMatFile, ms.confusion_matrix(goldAllLabs, predAllLabs), delimiter=',')

    def save(self, fileName):
        np.savetxt(fileName + '.uni.model', self.uniWeights, delimiter=',')
        np.savetxt(fileName + '.bi.model', self.biWeights, delimiter=',')

    def load(self, fileName):
        self.uniWeights = np.loadtxt(fileName + '.uni.model', delimiter=',')
        self.biWeights = np.loadtxt(fileName + '.bi.model', delimiter=',')

print("Import successfully.")
