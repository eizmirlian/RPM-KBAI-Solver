import cv2
import numpy as np

class ProblemAnalysis:
    #This class does big picture analysis on the problem rather than focusing on individual frames, will assume that the problem is a 3x3
    def __init__(self, figFrames):
        self.figFrames = figFrames
        self.problemImages = {}
        self.answerImages = {}
        for f in self.figFrames.keys():
            image = cv2.imread(figFrames[f].image)
            imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, threshold = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
            if str.isalpha(f):
                self.problemImages[f] = threshold
            else:
                self.answerImages[f] = threshold


    def bitwiseOp(self, operation, tol):
        if operation in ['areaAdd']:
            return self.areaOps(operation, tol)
        first = ['A', 'D', 'G']
        second = ['B', 'E', 'H']
        last = ['C', 'F']
        lastOptions = range(1, 9)
        i = 0
        while i < 2:
            iFirst = first[i]
            im1 = self.problemImages[iFirst].copy()
            iSecond = second[i]
            im2 = self.problemImages[iSecond].copy()
            iThird = last[i]
            im3 = self.problemImages[iThird].copy()
            outputIm = None
            not1 = cv2.bitwise_not(im1)
            not2 = cv2.bitwise_not(im2)
            if operation == 'and':  
                outputIm = cv2.bitwise_and(not1, not2)
            elif operation == 'xor':
                outputIm = cv2.bitwise_xor(not1, not2)
            elif operation == 'or':
                outputIm = cv2.bitwise_or(not1, not2)
            elif operation == 'add':
                outputIm = cv2.add(not1, not2)
            elif operation == 'sub':
                outputIm = cv2.subtract(not2, not1)
            outputIm = cv2.bitwise_not(outputIm)
            diffIm1 = cv2.subtract(outputIm, im3)
            diffIm2 = cv2.subtract(im3, outputIm)
            diff = max(cv2.countNonZero(diffIm1), cv2.countNonZero(diffIm2))
            if diff > tol:
                print(operation, diff)
                return False, None, None
            i += 1
        iFirst = first[2]
        im1 = self.problemImages[iFirst].copy()
        iSecond = second[2]
        im2 = self.problemImages[iSecond].copy()
        opIm = None
        not1 = cv2.bitwise_not(im1)
        not2 = cv2.bitwise_not(im2)
        if operation == 'and':  
            opIm = cv2.bitwise_and(not1, not2)
        elif operation == 'xor':
            opIm = cv2.bitwise_xor(not1, not2)
        elif operation == 'or':
            opIm = cv2.bitwise_or(not1, not2)
        elif operation == 'add':
            opIm = cv2.add(not1, not2)
        elif operation == 'sub':
            opIm = cv2.subtract(not2, not1)
        opIm = cv2.bitwise_not(opIm)
        bestDiff = tol
        bestAnswer = None
        for i in lastOptions:
            testIm = self.answerImages[str(i)]
            diffIm1 = cv2.subtract(opIm, testIm)
            diffIm2 = cv2.subtract(testIm, opIm)
            diff = max(cv2.countNonZero(diffIm1), cv2.countNonZero(diffIm2))
            if diff < bestDiff:
                bestAnswer = i
                bestDiff = diff
        print(operation, bestDiff, bestAnswer)
        if bestAnswer == None:
            return False, None, None
        return True, bestAnswer, bestDiff


    def areaOps(self, operation, tol):
        first = ['A', 'D', 'G']
        second = ['B', 'E', 'H']
        last = ['C', 'F']
        lastOptions = range(1, 9)
        i = 0
        while i < 2:
            iFirst = first[i]
            im1 = self.problemImages[iFirst].copy()
            not1 = cv2.bitwise_not(im1)
            iSecond = second[i]
            im2 = self.problemImages[iSecond].copy()
            not2 = cv2.bitwise_not(im2)
            iThird = last[i]
            im3 = self.problemImages[iThird].copy()
            not3 = cv2.bitwise_not(im3)
            diff = self.areaAddCheck(not1, not2, not3)
            if diff > tol:
                return False, None, None
            i += 1
        iFirst = first[2]
        im1 = self.problemImages[iFirst].copy()
        not1 = cv2.bitwise_not(im1)
        iSecond = second[2]
        im2 = self.problemImages[iSecond].copy()
        not2 = cv2.bitwise_not(im2)
        bestDiff = tol
        bestAnswer = None
        for i in lastOptions:
            testIm = self.answerImages[str(i)]
            notTest = cv2.bitwise_not(testIm)
            diff = self.areaAddCheck(not1, not2, notTest)
            if diff < bestDiff:
                bestAnswer = i
                bestDiff = diff
        print(operation, bestDiff, bestAnswer)
        if bestAnswer == None:
            return False, None, None
        return True, bestAnswer, bestDiff


    def areaAddCheck(self, im1, im2, im3):
        ar1 = cv2.countNonZero(im1)
        ar2 = cv2.countNonZero(im2)
        ar3 = cv2.countNonZero(im3)
        return np.abs(ar3 - (ar2 + (ar2 - ar1)))
