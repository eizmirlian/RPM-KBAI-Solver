# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
from PIL import Image
import numpy as np
import cv2
from BitAnalysis import ProblemAnalysis

class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        pass

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints 
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self, problem):
        problemType = problem.problemType
        if problemType == '2x2':
            solvingMethod = "ShapeOnly"
        else:
            solvingMethod = "Bit Pre-analysis"
        print(problem.name)
        if solvingMethod == "ShapeOnly":
            return self.shapeProblemSolvingStrategy(problem)
        if solvingMethod == "Bit Pre-analysis":
            return self.testBitAnalysis(problem, 700)


    def testBitAnalysis(self, problem, tol):
        figFrames = self.createFigFrames(problem)
        pbAnalysis = ProblemAnalysis(figFrames)
        bestDiff = tol
        bestAnswer = -1
        for op in [('and', tol), ('or', tol), ('xor', tol), ('add', tol), ('sub', tol), ('areaAdd', tol / 10)]:
            opMatch, opAnswer, opDiff = pbAnalysis.bitwiseOp(op[0], op[1])
            if opMatch and opDiff < bestDiff:
                bestDiff = opDiff
                bestAnswer = opAnswer
        if bestAnswer != -1:
            return bestAnswer
        else:
            return self.shapeProblemSolvingStrategy(problem)



    def shapeProblemSolvingStrategy(self, problem):

        bestAnswer = 0
        bestScore = None

        figFrames = self.createFigFrames(problem)
        is2by2 = problem.problemType == "2x2"
        figureA = figFrames["A"]
        figureB = figFrames["B"]
        figureC = figFrames["C"]
        if not is2by2:
                figureD = figFrames["D"]
                figureE = figFrames["E"]
                figureF = figFrames["F"]
                figureG = figFrames["G"]
                figureH = figFrames["H"]
        figureA.linkedFigures.extend([figureB, figureC])

        horizontal = []
        vertical = []
        
        if is2by2:
            hFigureComp = self.compareFigures(figureA, figureB, True)
            print("AB", hFigureComp)
            horizontal.append(hFigureComp)
            vFigureComp = self.compareFigures(figureA, figureC, False)
            print("AC", vFigureComp)
            vertical.append(vFigureComp)

        else:
            #print("AB", self.compareFigures(figureA, figureB))
            horizontal.append(self.compareFigures(figureA, figureB, True))
            #print("BC", self.compareFigures(figureB, figureC))
            horizontal.append(self.compareFigures(figureB, figureC, True))
            #print("DE", self.compareFigures(figureD, figureE))
            horizontal.append(self.compareFigures(figureD, figureE, True))
            #print("EF", self.compareFigures(figureE, figureF))
            horizontal.append(self.compareFigures(figureE, figureF, True))
            #print("GH", self.compareFigures(figureG, figureH))
            horizontal.append(self.compareFigures(figureG, figureH, True))
            vertical.append(self.compareFigures(figureA, figureD, False))
            vertical.append(self.compareFigures(figureB, figureE, False))
            vertical.append(self.compareFigures(figureC, figureF, False))
            vertical.append(self.compareFigures(figureD, figureG, False))
            vertical.append(self.compareFigures(figureE, figureH, False))
        
        if is2by2:
            numAnswerChoices = 6
        else:
            numAnswerChoices = 8
        i = 1
        while i <= numAnswerChoices:
            print(i)
            curr_score = 0
            answerChoice = figFrames[str(i)]
            if is2by2:
                testHTransforms = self.compareFigures(figureC, answerChoice, True)
                print("C ", testHTransforms)
                testVTransforms = self.compareFigures(figureB, answerChoice, False)
                print("B ", testVTransforms)
                others = [[figureA, figureB], [figureA, figureC]]
                prev = [figureC, figureB]
            else:
                testHTransforms = self.compareFigures(figureH, answerChoice, True)
                print(testHTransforms)
                testVTransforms = self.compareFigures(figureF, answerChoice, False)
                others = [[figureA, figureB, figureC], [figureD, figureE, figureF], [figureA, figureD, figureG], [figureB, figureE, figureH]]
                prev = [figureH, figureH, figureF, figureF]
            for o in range(len(prev)):
                multInc, score = answerChoice.compareMultiplicities(others[o], prev[o])
                if multInc:
                    #print(i)
                    curr_score += score
            for hTransforms in horizontal:
                for transform in hTransforms.keys():
                    if transform in testHTransforms.keys():
                        for quantity in testHTransforms[transform]:
                            comp_val = self.compareTransforms(transform, hTransforms[transform], quantity)
                            if comp_val > 0:
                                curr_score += comp_val
                            else:
                                curr_score -= 1
                    else:
                        curr_score -= len(hTransforms[transform])
            for vTransforms in vertical:
                for transform in vTransforms.keys():
                    if transform in testVTransforms.keys():
                        for quantity in testVTransforms[transform]:
                            comp_val = self.compareTransforms(transform, vTransforms[transform], quantity)
                            if comp_val > 0:
                                curr_score += comp_val
                            else:
                                curr_score -= 1
                    else:
                        curr_score -= len(vTransforms[transform])
            print(i, curr_score)
            if bestScore == None or curr_score > bestScore:
                bestAnswer = i
                bestScore = curr_score
            
            i += 1

        if bestAnswer != None:
            return bestAnswer
        else:
            return 3
                   




    def createFigFrames(self, problem):
        figFrames = {}
        figures = problem.figures
        for f in figures.keys():
            fig = FigureFrame(self.identifyShapes(figures[f]), {}, [], figures[f].visualFilename)
            figFrames[f] = fig
        return figFrames
            



    def compareFigures(self, figure1, figure2, horizontal):
        relatedIndex = 1
        if horizontal:
            relatedIndex = 0
        chosenTransforms = {}
        for shape in figure1.shapes:
            bestScore = 0
            mostSimilar = None
            closestMatch = None
            for other in figure2.shapes:
                transforms, similarityScore = shape.compare(other)
                if len(transforms.keys()) > 0 and similarityScore > bestScore and (other.related[relatedIndex] == None or other.related[relatedIndex][2] < similarityScore):
                    bestScore = similarityScore
                    mostSimilar = transforms
                    closestMatch = other
            if mostSimilar != None:
                for toAdd in mostSimilar.keys():
                    if toAdd not in chosenTransforms.keys():
                        chosenTransforms[toAdd] = [[mostSimilar[toAdd], bestScore]]
                    else:
                        chosenTransforms[toAdd].append([mostSimilar[toAdd], bestScore])
                closestMatch.related[relatedIndex] = [shape, mostSimilar, bestScore]
            else:
                if "Deleted" in chosenTransforms:
                    chosenTransforms["Deleted"].append([True, .1])
                else:
                    chosenTransforms["Deleted"] = [[True, .1]]
        
        return chosenTransforms




    def identifyShapes(self, figure):
        print(figure.name)
        shapes = []
        image = cv2.imread(figure.visualFilename)
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #blur = cv2.blur(imgray, (10,10))
        _, threshold = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
        notted = cv2.bitwise_not(threshold)
        #threshold = cv2.erode(imgray, np.ones((15, 15), np.uint8))
        whiteContours, _ = cv2.findContours(notted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        blackContours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = []
        contours.extend(blackContours)
        contours.extend(whiteContours)
        i = 0
        for contour in contours:

            approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
            #print(approx)
            numCorners = len(approx)
            if numCorners > 50:
                numCorners = -1
            area = cv2.contourArea(contour)
            if area == 0:
                continue
            M = cv2.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            center = (cx, cy)
            if imgray[cx, cy] > 150:
                fill = False
            else:
                fill = True

            closestCorner = self.closestCorner(contour, center)
            angle = self.findOrientation(closestCorner, center)

            
            newShape = ShapeFrame(center, area, angle, numCorners, fill)
            if not self.eliminateDuplicates(shapes, newShape):
                print(center, area, angle, numCorners, fill)
                shapes.append(newShape)

        return shapes
    

    def eliminateDuplicates(self, shapes, newShape):
        for shape in shapes:
            if shape.center == newShape.center and shape.fill == newShape.fill and np.abs(shape.size - newShape.size) < shape.size / 5 and np.abs(shape.corners - newShape.corners) < shape.corners / 5:
                return True
        if newShape.center == (91, 91) and newShape.size == 33489.0 and newShape.corners == 4:
                return True
        return False


    def closestCorner(self, contour, center):
        minCorner = None
        minDistance = -1
        for corner in contour:
            distance = np.sqrt(np.square(corner[0, 0] - center[0]) + np.square(corner[0, 1] - center[1]))
            if (minDistance == -1 or distance < minDistance) and distance > .1:
                minDistance = distance
                minCorner = corner
        return minCorner
    



    def findOrientation(self, closestCorner, center):
        vector = np.array([closestCorner[0, 0] - center[0], closestCorner[0, 1] - center[1]])
        if np.linalg.norm(vector) == 0:
            return 0
        norm = vector / np.linalg.norm(vector)
        if norm[0] == 0:
            angle = np.arcsin(norm[1])
        else:
            angle = np.arctan(norm[1] / norm[0])
        #print(angle)
        return angle




    def compareTransforms(self, t, transform, testTransform):
        toReturn = 0
        for actual in transform:
            if t == "Shrink":
                if np.abs(np.abs(actual[0]) - np.abs(testTransform[0])) < .3:
                    toReturn += 1
            elif t == "Expand":
                if np.abs(np.abs(actual[0]) - np.abs(testTransform[0])) < .3:
                    toReturn += 1
            elif t == "Rotate":
                if np.abs(np.abs(actual[0]) - np.abs(testTransform[0])) < .01:
                    toReturn += (actual[1] - np.abs(actual[1] - testTransform[1])) / actual[1]
            elif t == "Translate":
                if np.abs(actual[0][0] - testTransform[0][0]) + np.abs(actual[0][1] - testTransform[0][1]) < 20:
                    toReturn += (actual[1] - np.abs(actual[1] - testTransform[1])) / actual[1]
            elif t == "Shape Transform":
                if np.abs(actual[0] - testTransform[0]) < 10:
                    toReturn += 1
            elif t == "Same":
                if actual[0] == testTransform[0]:
                    toReturn += (actual[1] - np.abs(actual[1] - testTransform[1]))
            elif t == "Unclear":
                if actual[0] == testTransform[0]:
                    toReturn += 1
        return toReturn
        


class ShapeFrame:
    # Class representing a frame for a shape
    def __init__(self, center, size, orientation, numCorners, fill):
        self.center = center
        self.size = size
        self.orientation = orientation
        self.corners = numCorners
        self.fill = fill
        self.related = [None, None]

    def compare(self, other):
        centerMatch = False
        sizeMatch = False
        orientMatch = False
        cornersMatch = False
        fillMatch = False
        similarityScore = 0
        if np.abs(other.center[0] - self.center[0]) + np.abs(other.center[1] - self.center[1]) <= 15:
            centerMatch = True
            similarityScore += 5
        if np.abs((other.size / self.size) - 1) <= .1:
            sizeMatch = True
            similarityScore += 6
        if np.abs(other.orientation - self.orientation) <= .1:
            orientMatch = True
            similarityScore += 4
        if other.corners == self.corners:
            cornersMatch = True
            similarityScore += 6
        if self.fill == other.fill:
            fillMatch = True
            similarityScore += 1
        if similarityScore > 3:
            #print("Match between " + str(self.corners) + " and " + str(other.corners), similarityScore)
            return self.inferTransformation(centerMatch, sizeMatch, orientMatch, cornersMatch, fillMatch, other), similarityScore
        else:
            return {}, similarityScore

    def inferTransformation(self, centerMatch, sizeMatch, orientMatch, cornersMatch, fillMatch, other):
        transforms = {}
        if cornersMatch and not sizeMatch:
            if self.size > other.size:
                transforms["Shrink"] = other.size / self.size
                #print("S")
            else:
                transforms["Expand"] = other.size / self.size
                #print("E")
        if cornersMatch and not centerMatch:
            #print("T")
            transforms["Translate"] = (other.center[0] - self.center[0], other.center[1] - self.center[1])
        if not orientMatch:
            #print("R", other.orientation - self.orientation)
            transforms["Rotate"] = other.orientation - self.orientation
        if not cornersMatch and centerMatch and sizeMatch:
            #print("ST")
            transforms["Shape Transform"] = other.corners - self.corners
        if cornersMatch and not fillMatch:
            #print("FT")
            transforms["Fill Toggle"] = True
        if cornersMatch and centerMatch and sizeMatch and orientMatch and fillMatch:
            #print("Same")
            transforms["Same"] = True
        if len(transforms.keys()) == 0:
            transforms["Unclear"] = True
        return transforms

class FigureFrame:

    def __init__(self, shapes, transformations, linkedFigures, image):
        self.image = image
        self.shapes = shapes
        self.transformations = transformations
        self.linkedFigures = linkedFigures

    def compareMultiplicities(self, others, preceding):
        multMatch = True
        for o in others:
            if len(o.shapes) != len(self.shapes):
                multMatch = False
        if not multMatch:
            increment = None
            prev = None
            validIncrement = True
            for o in others:
                if prev == None:
                    prev = o
                elif increment == None:
                    increment = len(o.shapes) - len(prev.shapes)
                else:
                    if increment != len(o.shapes) - len(prev.shapes):
                        validIncrement = False
            if validIncrement:
                print(increment)
                if increment == len(self.shapes) - len(preceding.shapes):
                    return True, 5
            return False, 0
        return True, 0

