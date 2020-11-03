#  Author: Diego Tapia R. J
#  E-mail: root.chile@gmail.com - diego.tapia.r@mail.pucv.cl

import time
import numpy as np
import math
from scipy import special as scyesp

"""
Realiza una transformación de una matrix en el dominio de los continuos a una matriz binaria.
Usada para discretización de soluciones en metaheurísticas. 
Donde cada fila es una solución (individuo de la población), y las columnas de la matriz corresponden a las dimensiones de la solución.
Params
----------
matrixCont : matriz de continuos.
matrixBin  : matriz de poblacion binarizada en t-1. 
bestRowBin : individuo (fila) de mejor fitness.
SolutionRanking: Lista de indices ordenadas por fitness, en la posición 0 esta el best
transferFunction: Funciones de transferencia (V1,..,V4, S1,..,S4)
binarizationOperator: Operador de binarización (Standard,Complement, Elitist, Static, Roulette)
Returns
-------
matrixBinOut: matriz binaria.
Definiciones
---------
Para t=0, es decir, si matrixBin es vacía, se utiliza por defecto binarizationOperator = Standard.
"""

class Discretization:
    
    def __init__(self, matrixProbT, matrixBin, SolutionRanking, discretizationOperator):

        self.discretizationOperator = discretizationOperator
        self.matrixProbT = matrixProbT

        self.matrixBin = np.ndarray(matrixBin.shape, dtype=float, buffer=matrixBin)
        self.SolutionRanking = SolutionRanking
        self.bestRow = SolutionRanking[0]
        #output
        self.matrixBinOut = np.zeros(self.matrixBin.shape)


    #Binarization
    def B_Standard(self):
        matrixRand = np.random.uniform(low=0.0,high=1.0,size=self.matrixBin.shape)
        self.matrixBinOut = np.greater(self.matrixProbT,matrixRand).astype(int)

    def B_Complement(self):
        matrixRand = np.random.uniform(low=0.0,high=1.0,size=self.matrixBin.shape)
        matrixComplement = np.abs(1-self.matrixBin)
        self.matrixBinOut = np.multiply(np.greater_equal(self.matrixProbT,matrixRand).astype(int),matrixComplement)

    def B_Elitist(self):
        matrixRand = np.random.uniform(low=0.0,high=1.0,size=self.matrixBin.shape)
        conditionMatrix = np.greater(self.matrixProbT,matrixRand)
        bestIndividual = self.matrixBin[self.bestRow]
        self.matrixBinOut = np.where(conditionMatrix==True,bestIndividual,0)

    def B_Static(self):
        alfa = 1/3        
        self.matrixBinOut[self.matrixProbT<=alfa] = 0
        self.matrixBinOut[(self.matrixProbT > alfa) & (self.matrixProbT <= 0.5*(1+alfa))] = self.matrixBin[(self.matrixProbT > alfa) & (self.matrixProbT <= 0.5*(1+alfa))]
        self.matrixBinOut[self.matrixProbT>=0.5*(1+alfa)] = 1

    def B_ElitistRoulette(self):
        matrixRand = np.random.uniform(low=0.0,high=1.0,size=self.matrixBin.shape)
        conditionMatrix = np.greater(self.matrixProbT,matrixRand)

        alfa = 0.2
        if self.SolutionRanking.shape[0]*alfa < 1:
            bestIndividual = self.matrixBin[self.SolutionRanking[0]]
        else:
            BestSolutionRaking = int(self.SolutionRanking.shape[0]*alfa)
            random = np.random.randint(low = 0, high = BestSolutionRaking)
            bestIndividual = self.matrixBin[self.SolutionRanking[random]]

        self.matrixBinOut = np.where(conditionMatrix==True,bestIndividual,0)

    def discretiza(self):

        if self.discretizationOperator == 'Standard':
            self.B_Standard()

        if self.discretizationOperator == 'Complement':
            self.B_Complement()

        if self.discretizationOperator == 'Elitist':
            self.B_Elitist()

        if self.discretizationOperator == 'Static':
            self.B_Static()

        if self.discretizationOperator == 'ElitistRoulette':
            self.B_ElitistRoulette()

        return self.matrixBinOut
