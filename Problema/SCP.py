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
from Problema.scp.repair import ReparaStrategy as _repara
from datetime import datetime
#import multiprocessing as mp
#from numpy.random import default_rng
from Problema.scp.repair import cumpleRestricciones as reparaGPU
from Problema.scp.permutationRank import PermRank
from Problema.Problema import Problema


class SCP(Problema):
    def __init__(self, instancePath = None):
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
        self.tTransferencia = "sShape2"
        self.tBinary = "Standar"        
        self.binarizationStrategy = _binarization.BinarizationStrategy(self.tTransferencia, self.tBinary)        
        self.repair = _repara.ReparaStrategy(self.instance.get_r()
                                            ,self.instance.get_c()
                                            ,self.instance.get_rows()
                                            ,self.instance.get_columns())
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
        return [-10,1]

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
    
    def getParametros(self):
        return self.parametros

    def evaluarFitness(self, soluciones):
        evaluaciones = self.evalObj(soluciones)
        mejorEvaluacion = np.min(evaluaciones)
        if self.mejorEvaluacion is None: self.mejorEvaluacion = mejorEvaluacion
        idxMejorEval = evaluaciones == mejorEvaluacion
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
        
        return evaluaciones

    def getIndsMejora(self, f1, f2):
        #cuanto mejora f2 a f1 
        assert f1.shape == f2.shape, f"Fitness 1 {f1.shape} diferente a fitness 2 {f2.shape}"
        return (f1-f2)/f1

    def getMejorIdx(self, fitness):
        return np.argmin(fitness)

    def getPeorIdx(self, fitness):
        return np.argmax(fitness)

    def eval(self, encodedInstance):
        decoded, numReparaciones = self.frepara(encodedInstance)
        fitness = self.evalInstance(encodedInstance)
        return fitness, decoded, numReparaciones


    def evalEnc(self, encodedInstance):
        decoded, numReparaciones = self.decodeInstance(encodedInstance)
        fitness = self.evalInstance(decoded)
        if self.mejorFitness is None or fitness > self.mejorFitness:
            self.mejorFitness = fitness
            self.binarizationStrategy.mejorSol = decoded
        encoded = self.encodeInstance(decoded)
        return fitness, decoded, numReparaciones,encoded

    def evalEncBatch(self, encodedInstances):
        inicio = datetime.now()
        decoded, numReparaciones = self.decodeInstancesBatch(encodedInstances)
        fin = datetime.now()
        #print(f"decoding demoro {fin-inicio}")
        #print(f"evalEncBatch {decoded.shape}")
        #exit()
        fitness = self.evalInstanceBatch(decoded)
        
        #encoded = self.encodeInstanceBatch(decoded)
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
        start = datetime.now()
        b = np.array([self.binarizationStrategy.binarize(inst) for inst in encodedInstances])
        #print(encodedInstances)
        encodedInstances = np.array(b)
        #encodedInstances = np.array(encodedInstances)
        #print(encodedInstances.shape)
        #exit()
        #print(f"encoded instances: {encodedInstances.shape}")
        #b = np.array([self.decodeInstance(encodedInstances[i,:])[0] for i in range(encodedInstances.shape[0])])
        #print(f"discretizado: {b.shape}")
        #exit()
        end = datetime.now()

        binTime = end-start
        numReparaciones = 0
        
        repaired = self.freparaBatch(b)
        return repaired, numReparaciones
    
    
    def decodeInstance(self, encodedInstance):
        encodedInstance = np.array(encodedInstance).astype(np.int8)
        if encodedInstance.shape[0] != self.particiones.shape[0]:
            raise Exception("La instancia encodeada cambio su tamaño")

        binario = []
        #print(encodedInstance)
        #raise Exception
        for idx in range(encodedInstance.shape[0]):
            #print(f"self.particiones[idx], encodedInstance[idx] {self.particiones[idx]}, {encodedInstance[idx]}")
            binario.extend(self.permRank.unrank(self.particiones[idx], encodedInstance[idx]).tolist())
        b = np.array(binario)
        

        #b = self.binarizationStrategy.binarize(encodedInstance)
        numReparaciones = 0
        #if not self.penalizar:
        #        b, numReparaciones = self.frepara(b)
        return b, numReparaciones
        
    def binarize(self, x):
        return _binarization.BinarizationStrategy(x,self.tTransferencia, self.tBinary)
   
#    @profile
    def evalInstance(self, decoded):
        return -(self.fObj(decoded, self.instance.get_c())) if self.repair.cumple(decoded) == 1 else -1000000
    
    def evalInstanceBatch(self, decoded):
        start = datetime.now()
        ret = np.sum(np.array(self.instance.get_c())*decoded, axis=1)
        end = datetime.now()
        return ret
    
#    @profile
    def fObj(self, pos,costo):
        return np.sum(np.array(pos) * np.array(costo))
  
#    @profile
    def freparaBatch(self,x):
        start = datetime.now()
        #print(f"freparaBatch {x.shape}")
        #exit()
        reparadas = reparaGPU.reparaSoluciones(x, self.instance.get_r(), self.instance.get_c(), self.instance.pondRestricciones)
        #print(reparadas.shape)
        #exit()
        end = datetime.now()
        #print(f"reparacion demoro {end-start}")
        return reparadas
    
    
    def frepara(self,x):
        start = datetime.now()
        cumpleTodas=0
        cumpleTodas=self.repair.cumple(x)
        if cumpleTodas == 1: return x, 0
        
        x, numReparaciones = self.repair.repara_one(x)    
        x = self.mejoraSolucion(x)
        end = datetime.now()
        return x, numReparaciones
    
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
#        args = []
        #mejorSol = None
        if self.mejoresSoluciones is None:
            args = np.zeros((numSols, self.getNumDim()), dtype=np.float)
            #args = np.ones((numSols, self.getNumDim()), dtype=np.float)
            #args = np.random.randint(low=self.getRangoSolucion()['min'], high=self.getRangoSolucion()['max'], size=(numSols,self.getRangoSolucion()['max'].shape[0]))
            #print(args)
            #exit()
        else:
            #self.mejorSolHist = (mejorSol+self.mejorSolHist)/2
            args = []
            for i in range(numSols):
                idx = np.random.randint(low=0, high=self.mejoresSoluciones.shape[0])
                sol = self.mejoresSoluciones[idx].copy()
                #idx = np.random.randint(low=0, high=sol.shape[0])
                #print(np.argwhere(sol > 0).reshape(-1))
                #exit()
                idx = np.random.choice(np.argwhere(sol > np.mean(sol)*1.5).reshape(-1), 1)[0]
                #print(idx)
                #exit()
                
                sol[idx] += np.random.randint(low=-10, high=-1)
                #if sol[idx] > self.particiones[idx]: sol[idx] = self.particiones[idx]
                #if sol[idx] < 0: sol[idx] = 0
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
        