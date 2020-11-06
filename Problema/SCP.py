#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 22:02:44 2019

@author: mauri
"""
import sys
#sys.path.insert(1, 'C:\\Users\\mauri\\proyectos\\GPUSCPRepair\\cudaTest\\reparacionGpu')

import numpy as np
from Problema.scp import read_instance as r_instance
from Problema.scp import binarizationstrategy as _binarization
from Utils.BinarizationScheme import BinarizationScheme as BS
from Problema.scp.repair import ReparaStrategy as _repara
from datetime import datetime
#import multiprocessing as mp
#from numpy.random import default_rng
from Problema.scp.repair import cumpleRestricciones as reparaGPU
from Problema.scp.permutationRank import PermRank
from Problema.Problema import Problema



class SCP(Problema):
    def __init__(self, instancePath):
#        print(f'LEYENDO INSTANCIA')
        self.mejorEvaluacion = None
        self.mejoresSoluciones = None
        self.mejorEvaluacion = None
        self.parametros = {}
        self.instancia = instancePath
        self.instance = r_instance.Read(instancePath)
        self.optimo = self.instance.optimo
#        print(f'FIN LEYENDO INSTANCIA')
        if(self.instance.columns != np.array(self.instance.get_c()).shape[0]):
            raise Exception(f'self.instance.columns {self.instance.columns} != np.array(self.instance.get_c()).shape[1] {np.array(self.instance.get_c()).shape[1]})')
        self.repair = _repara.ReparaStrategy(self.instance.get_r()
                                    ,self.instance.get_c()
                                    ,self.instance.get_rows()
                                    ,self.instance.get_columns())
        #Son necesarios para la Binarización
        self.BinarizationScheme = None
        self.matrixBin = None      
        self.solutionsRanking = None
        self.evaluaciones = None
        
        self.paralelo = False
        self.penalizar = False
        self.mejorSolHist = np.ones((self.instance.get_columns())) * 0.5
        self.mejorFitness = None
        self.partSize = 8
        self.rangeMax = []
        self.permRank = PermRank()
        self.particiones = []
        for _ in range(int(self.instance.get_columns()/self.partSize)):
            self.rangeMax.append(self.permRank.totalPerm(self.partSize))
            self.particiones.append(self.partSize)

        if self.instance.get_columns()%self.partSize > 0:
            self.rangeMax.append(self.permRank.totalPerm(self.instance.get_columns()%self.partSize))
            self.particiones.append(self.instance.get_columns()%self.partSize)
        self.rangeMax = np.array(self.rangeMax)
        self.particiones = np.array(self.particiones)


    def getNombre(self):
        return 'SCP'
    
    def getNumDim(self):
        return self.instance.columns
        #return self.particiones.shape[0]

    def getRangoSolucion(self):
        return {'max': self.rangeMax, 'min':np.zeros(self.rangeMax.shape[0])}

    def getDominioDim(self):
        return [-10,10]

    def evalObj(self, soluciones):
        decoded, _ = self.decodeInstancesBatch(soluciones)
        return self.evalInstanceBatch(decoded)

    def getIndiceMejora(self):
        return self.indiceMejora

    def getMejorEvaluacion(self):
        return self.mejorEvaluacion

    def setParametros(self, parametros):
        for parametro in parametros:
            self.parametros[parametro] = parametros[parametro]
        self.tTransferencia = self.parametros[SCP.TRANSFER_FUNCTION]
        self.tBinary = self.parametros[SCP.BINARIZATION]
        self.BinarizationScheme = BS()
        
        #self.binarizationStrategy = _binarization.BinarizationStrategy(self.tTransferencia, self.tBinary)      
        
        self.repairType = self.parametros[SCP.REPAIR]  
    
    def getParametros(self):
        return self.parametros

    def evaluarFitness(self, soluciones):
        self.evaluaciones = self.evalObj(soluciones)
        mejorEvaluacion = np.min(self.evaluaciones)
        if self.mejorEvaluacion is None: self.mejorEvaluacion = mejorEvaluacion
        idxMejorEval = self.evaluaciones == mejorEvaluacion
        mejoresSoluciones = np.unique(soluciones[idxMejorEval], axis=0)
        self.indiceMejora = self.getIndsMejora(self.mejorEvaluacion,mejorEvaluacion)
        if mejorEvaluacion < self.mejorEvaluacion:
            self.mejoresSoluciones = mejoresSoluciones
            self.mejorEvaluacion = mejorEvaluacion
        if mejorEvaluacion == self.mejorEvaluacion:
            mejoresSolucionesL = list(mejoresSoluciones)
            if self.mejoresSoluciones is not None:
                mejoresSolucionesL.extend(list(self.mejoresSoluciones))
            self.mejoresSoluciones = np.unique(np.array(mejoresSolucionesL), axis=0)
        
        return self.evaluaciones

    def getIndsMejora(self, f1, f2):
        #cuanto mejora f2 a f1 
        assert f1.shape == f2.shape, f"Fitness 1 {f1.shape} diferente a fitness 2 {f2.shape}"
        return (f1-f2)/f1

    def getMejorIdx(self, fitness):
        return np.argmin(fitness)

    def getPeorIdx(self, fitness):
        return np.argmax(fitness)

    # def eval(self, encodedInstance):
    #     decoded, numReparaciones = self.frepara(encodedInstance)
    #     fitness = self.evalInstance(encodedInstance)
    #     return fitness, decoded, numReparaciones

    def evalEnc(self, encodedInstance):
        decoded, numReparaciones = self.decodeInstance(encodedInstance)
        fitness = self.evalInstance(decoded)
        if self.mejorFitness is None or fitness > self.mejorFitness:
            self.mejorFitness = fitness
        encoded = self.encodeInstance(decoded)
        return fitness, decoded, numReparaciones,encoded

    def evalEncBatch(self, encodedInstances):

        decoded, numReparaciones = self.decodeInstancesBatch(encodedInstances)
        fitness = self.evalInstanceBatch(decoded)
        
        encoded = decoded.astype(float)
        return fitness, decoded, numReparaciones, encoded
    
    def evalDecBatch(self, encodedInstances, mejorSol):
        fitness = self.evalInstanceBatch(encodedInstances)
        
        
        return fitness, encodedInstances, None
    
    def encodeInstanceBatch(self, decodedInstances):
        ret = np.array([self.encodeInstance(decodedInstances[i]) for i in range(decodedInstances.shape[0])],dtype=float)
        return ret

    def encodeInstance(self, decodedInstance):
        currIdx = 0
        res = []
        for partSize in self.particiones:
            res.append(self.permRank.getRank(decodedInstance[currIdx:currIdx+partSize]))
            currIdx+=partSize
        return np.array(res)

#    @profile
        
    def decodeInstancesBatch(self, encodedInstances):

        if self.matrixBin is None:
            self.matrixBin = np.zeros(encodedInstances.shape)
        if self.solutionsRanking is None:
            self.solutionsRanking = np.zeros(encodedInstances.shape[0], dtype = np.int8)
        else:
            self.solutionsRanking =  np.argsort(self.evaluaciones)

        #b = np.array([self.binarizationStrategy.binarize(inst) for inst in encodedInstances])
        self.matrixBin = self.BinarizationScheme.TwoSteps(encodedInstances, self.matrixBin, self.solutionsRanking, self.tTransferencia, self.tBinary)

        numReparaciones = 0
        repaired,numReparaciones = self.freparaBatch(self.matrixBin)
        return repaired, numReparaciones
   
    
    def decodeInstance(self, encodedInstance):
        encodedInstance = np.array(encodedInstance).astype(np.int8)
        if encodedInstance.shape[0] != self.particiones.shape[0]:
            raise Exception("La instancia encodeada cambio su tamaño")

        binario = []
        for idx in range(encodedInstance.shape[0]):
            binario.extend(self.permRank.unrank(self.particiones[idx], encodedInstance[idx]).tolist())
        b = np.array(binario)
        

        numReparaciones = 0
        return b, numReparaciones
        
    # def binarize(self, x):
    #     return _binarization.BinarizationStrategy(x,self.tTransferencia, self.tBinary)
   
#    @profile
    def evalInstance(self, decoded):
        return -(self.fObj(decoded, self.instance.get_c())) if self.repair.cumple(decoded) == 1 else -1000000
    
    def evalInstanceBatch(self, decoded):
        ret = np.sum(np.array(self.instance.get_c())*decoded, axis=1)
        return ret
    
#    @profile
    def fObj(self, pos,costo):
        return np.sum(np.array(pos) * np.array(costo))
  
#    @profile
    def freparaBatch(self,x):

        if self.repairType == "repairGPU":
            reparadas = reparaGPU.reparaSoluciones(x, self.instance.get_r(), self.instance.get_c(), self.instance.pondRestricciones)
            numReparaciones = 0
            return reparadas, numReparaciones
        elif self.repairType == "repairSimple" or self.repairType == "repairCompleja" :
            numReparaciones = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                cumpleTodas=0
                cumpleTodas=self.repair.cumple(x[i])
                if cumpleTodas == 0:
                    x[i], numReparaciones[i] = self.repair.repara_one(x[i],self.repairType)    
                    #x = self.mejoraSolucion(x)
            return x, numReparaciones
           
    # def frepara(self,x):
    #     cumpleTodas=0
    #     cumpleTodas=self.repair.cumple(x)
    #     if cumpleTodas == 1: return x, 0
        
    #     x, numReparaciones = self.repair.repara_one(x)    
    #     x = self.mejoraSolucion(x)
    #     return x, numReparaciones
    
    def mejoraSolucion(self, solucion):
        solucion = np.array(solucion)
        costos = solucion * self.instance.get_c()
        cosOrd = np.argsort(costos)[::-1]
        for pos in cosOrd:
            if costos[pos] == 0: break
            modificado = solucion.copy()
            modificado[pos] = 0
            if self.repair.cumple(modificado) == 1:
                solucion = modificado
        return solucion
    
    def generarSoluciones(self, numSols):

        if self.mejoresSoluciones is None:
            args = np.zeros((numSols, self.getNumDim()), dtype=np.float)
        else:
            args = []
            for i in range(numSols):
                idx = np.random.randint(low=0, high=self.mejoresSoluciones.shape[0])
                sol = self.mejoresSoluciones[idx].copy()
                idx = np.random.choice(np.argwhere(sol > np.mean(sol)*1.5).reshape(-1), 1)[0]
                
                sol[idx] += np.random.randint(low=-10, high=-1)
                args.append(sol)
            args = np.array(args)

        fitness = []
        ant = self.penalizar
        self.penalizar = False
        fitness, _, _, sol = self.evalEncBatch(args)
        return sol
    
    def graficarSol(self, datosNivel, parametros, nivel, id = 0):
        if not hasattr(self, 'graficador'):
            self.initGrafico()
        y = datosNivel['soluciones'][0]
        vels = datosNivel['velocidades'][0]
        self.graficador.live_plotter(np.arange(y.shape[0]),y, 'soluciones', dotSize=0.1, marker='.')
        self.graficador.live_plotter(np.arange(vels.shape[0]), vels, 'velocidades', dotSize=0.1, marker='.')
        self.graficador.live_plotter(np.arange(parametros.shape[0]), parametros, 'paramVel', dotSize=1.5, marker='-')
    
    def getMatrixBin(self):
        return self.matrixBin