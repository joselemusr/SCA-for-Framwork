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

class TransferFunction:
    
    def __init__(self, matrixCont, transferFunction):

        self.transferFunction = transferFunction

        self.matrixCont = np.ndarray(matrixCont.shape, dtype=float, buffer=matrixCont)

        #output
        self.matrixProbT = np.zeros(self.matrixCont.shape)

    #Funciones de Transferencia

    def T_V1(self):
        # revisado ok
        self.matrixProbT = np.abs(scyesp.erf(np.divide(np.sqrt(np.pi),2)*self.matrixCont))

    def T_V2(self):
        self.matrixProbT = np.abs(np.tanh(self.matrixCont))

    def T_V3(self):

        # for i in range(len(self.matrixCont)):
        #     for d in range(len(self.matrixCont[i])):
        #         self.matrixProbT[i][d] = math.fabs(self.matrixCont[i][d]/math.sqrt(1+(self.matrixCont[i][d]**2)))
        
        # print(self.matrixProbT)
        
        self.matrixProbT = np.abs(np.divide(self.matrixCont, np.sqrt(1+np.power(self.matrixCont,2))))

    def T_V4(self):

        # for i in range(len(self.matrixCont)):
        #     for d in range(len(self.matrixCont[i])):
        #         self.matrixProbTAux[i][d] = math.fabs((2/math.pi)*math.atan((math.pi/2)*self.matrixCont[i][d]))
        
        self.matrixProbT = np.abs( np.divide(2,np.pi)*np.arctan( np.divide(np.pi,2)*self.matrixCont ))

        # return (self.matrixProbTAux-self.matrixProbT)

    def T_S1(self):

        # for i in range(len(self.matrixCont)):
        #     for d in range(len(self.matrixCont[i])):
        #         self.matrixProbTAux[i][d] = 1 / (1 + math.exp(-2 * self.matrixCont[i][d]))

        self.matrixProbT = np.divide(1, ( 1 + np.exp(-2*self.matrixCont) ) )

        # return (self.matrixProbTAux-self.matrixProbT)

    def T_S2(self):

        # for i in range(len(self.matrixCont)):
        #     for d in range(len(self.matrixCont[i])):
        #         self.matrixProbTAux[i][d] = 1 / (1 + math.exp(-1 * self.matrixCont[i][d]))

        self.matrixProbT = np.divide(1, ( 1 + np.exp(-1*self.matrixCont) ) )

        # return (self.matrixProbTAux-self.matrixProbT)

    def T_S3(self):

        # for i in range(len(self.matrixCont)):
        #     for d in range(len(self.matrixCont[i])):
        #         self.matrixProbTAux[i][d] = 1 / (1 + math.exp(np.divide(-1*self.matrixCont[i][d],2)))

        self.matrixProbT = np.divide(1, ( 1 + np.exp(np.divide(-1*self.matrixCont,2) ) ))

        # return (self.matrixProbTAux-self.matrixProbT)

    def T_S4(self):

        # for i in range(len(self.matrixCont)):
        #     for d in range(len(self.matrixCont[i])):
        #         self.matrixProbTAux[i][d] = 1 / (1 + math.exp(np.divide(-1*self.matrixCont[i][d],3)))

        self.matrixProbT = np.divide(1, ( 1 + np.exp(np.divide(-1*self.matrixCont,3) ) ))

        # return (self.matrixProbTAux-self.matrixProbT)

    def transfiere(self):

        if self.transferFunction == 'V1':
            self.T_V1()

        if self.transferFunction == 'V2':
            self.T_V2()

        if self.transferFunction == 'V3':
            self.T_V3()

        if self.transferFunction == 'V4':
            self.T_V4()

        if self.transferFunction == 'S1':
            self.T_S1()

        if self.transferFunction == 'S2':
            self.T_S2()

        if self.transferFunction == 'S3':
            self.T_S3()

        if self.transferFunction == 'S4':
            self.T_S4()

        return self.matrixProbT
