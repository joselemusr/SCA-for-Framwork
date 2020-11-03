from . import TransferFunction as TF
from . import Discretization as D
import numpy as np

class BinarizationScheme:
    
    def TwoSteps(self, matrixCont, matrixBin, SolutionRanking, transferFunction, discretizationOperator):

        self.transferFunction = transferFunction
        self.discretizationOperator = discretizationOperator


        self.matrixCont = np.ndarray(matrixCont.shape, dtype=float, buffer=matrixCont)
        self.matrixBin = np.ndarray(matrixBin.shape, dtype=float, buffer=matrixBin)
        self.SolutionRanking = SolutionRanking
        self.bestRow = np.argmin(SolutionRanking) 

        #output
        self.matrixProbT = np.zeros(self.matrixCont.shape)
        self.matrixBinOut = np.zeros(self.matrixBin.shape)

        #TransferFunction
        transferFunctionObj = TF.TransferFunction(self.matrixCont,self.transferFunction)
        self.matrixProbT = transferFunctionObj.transfiere()

        #Discretization
        discretizationObj = D.Discretization(self.matrixProbT,self.matrixBin,self.SolutionRanking,self.discretizationOperator)
        self.matrixBinOut = discretizationObj.discretiza()
        
        return self.matrixBinOut





