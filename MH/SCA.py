from MH.Metaheuristica import Metaheuristica
import numpy as np
from DTO.IndicadoresMH import IndicadoresMH
from DTO import TipoIndicadoresMH
import math
from Utils import CalculoDeDiversidades as dv

class SCA(Metaheuristica):
    def __init__(self):
        self.problema = None
        self.soluciones = None
        self.parametros = {}
        self.idxMejorSolucion = None
        self.mejoraPorSol = None
        self.mejoraPorSolAcumulada = None
        self.mejorSolHistorica = None
        self.mejorFitHistorica = None
        self.fitnessAnterior = None
        self.IteracionActual = None
        self.maxDiversidades = None
        self.diversidades = None
        self.PorcentajeExplor = None
        self.PorcentajeExplot = None
        self.state = None
        print(f"Mh SCA creada")
        

    
    def setIteracionActual(self, IteracionActual):
        self.IteracionActual = IteracionActual

    def getIteracionActual(self):
        return self.IteracionActual

    def setProblema(self, problema):
        self.problema = problema

    def getProblema(self):
        return self.problema

    def generarPoblacion(self, numero):
        self.soluciones = self.problema.generarSoluciones(numero)
        fitness = self.problema.evaluarFitness(self.soluciones)
        matrixBin = self.problema.getMatrixBin()
        self.diversidades, self.maxDiversidades, self.PorcentajeExplor, self.PorcentajeExplot, self.state = dv.ObtenerDiversidadYEstado(matrixBin,self.getParametros()[SCA.MAXDIVERSIDADES])

        self.indicadores = {
            TipoIndicadoresMH.INDICE_MEJORA:self.problema.getIndiceMejora()
            ,TipoIndicadoresMH.FITNESS_MEJOR_GLOBAL:self.problema.getMejorEvaluacion()
            ,TipoIndicadoresMH.FITNESS_MEJOR_ITERACION:fitness[self.idxMejorSolucion]
            ,TipoIndicadoresMH.FITNESS_PROMEDIO:np.mean(fitness)
            ,TipoIndicadoresMH.DIVERSIDADES:self.diversidades
            ,TipoIndicadoresMH.PORCENTAJEEXPLORACION:self.PorcentajeExplor
            ,TipoIndicadoresMH.PORCENTAJEEXPLOTACION:self.PorcentajeExplot
            ,TipoIndicadoresMH.ESTADO:self.state
        }
    
    def setParametros(self, parametros):
        for parametro in parametros:
            self.parametros[parametro] = parametros[parametro]
    
    def getParametros(self):
        return self.parametros

    def realizarBusqueda(self):
        self._perturbarSoluciones()
        fitness = self.problema.evaluarFitness(self.soluciones)
        assert self.soluciones.shape[0] == fitness.shape[0], "El numero de fitness es diferente al numero de soluciones"
        if self.fitnessAnterior is None: self.fitnessAnterior = fitness
        self.idxMejorSolucion = self.problema.getMejorIdx(fitness)
        self.mejoraPorSol = self.problema.getIndsMejora(self.fitnessAnterior, fitness)
        assert self.soluciones.shape[0] == self.mejoraPorSol.shape[0], "El numero de indices de mejora es diferente al numero de soluciones"
        if self.mejoraPorSolAcumulada is None: self.mejoraPorSolAcumulada = np.zeros((self.soluciones.shape[0]))
        self.mejoraPorSolAcumulada += self.mejoraPorSol
        if self.mejorSolHistorica is None: self.mejorSolHistorica = self.soluciones
        if self.mejorFitHistorica is None: self.mejorFitHistorica = fitness

        mejorIdx = self.problema.getIndsMejora(self.mejorFitHistorica, fitness) > 0
        self.mejorSolHistorica[mejorIdx] = self.soluciones[mejorIdx]
        self.mejorFitHistorica[mejorIdx] = fitness[mejorIdx]


        self.fitnessAnterior = fitness

        matrixBin = self.problema.getMatrixBin()
        self.diversidades, self.maxDiversidades, self.PorcentajeExplor, self.PorcentajeExplot, self.state = dv.ObtenerDiversidadYEstado(matrixBin,self.maxDiversidades)

        self.indicadores = {
            TipoIndicadoresMH.INDICE_MEJORA:self.problema.getIndiceMejora()
            ,TipoIndicadoresMH.FITNESS_MEJOR_GLOBAL:self.problema.getMejorEvaluacion()
            ,TipoIndicadoresMH.FITNESS_MEJOR_ITERACION:fitness[self.idxMejorSolucion]
            ,TipoIndicadoresMH.FITNESS_PROMEDIO:np.mean(fitness)
            ,TipoIndicadoresMH.DIVERSIDADES:self.diversidades
            ,TipoIndicadoresMH.PORCENTAJEEXPLORACION:self.PorcentajeExplor
            ,TipoIndicadoresMH.PORCENTAJEEXPLOTACION:self.PorcentajeExplot
            ,TipoIndicadoresMH.ESTADO:self.state
        }

    def _perturbarSoluciones(self): 
        if self.idxMejorSolucion is None:
            self.idxMejorSolucion = np.random.randint(low = 0, high = self.soluciones.shape[0])

        r1 = self.parametros['a'] - (self.IteracionActual*(self.parametros['a']/int(self.getParametros()[SCA.NUM_ITER]))) #Escalar
        r2 = (2*np.pi) * np.random.uniform(low=0.0,high=1.0, size=self.soluciones.shape)
        r3 = np.random.uniform(low=0.0,high=2.0, size=self.soluciones.shape)
        r4 = np.random.uniform(low=0.0,high=1.0, size=self.soluciones.shape[0])
        self.soluciones[r4<0.5] = self.soluciones[r4<0.5] + np.multiply(r1,np.multiply(np.sin(r2[r4<0.5]),np.abs(np.multiply(r3[r4<0.5],self.soluciones[self.idxMejorSolucion])-self.soluciones[r4<0.5])))
        self.soluciones[r4>=0.5] = self.soluciones[r4>=0.5] + np.multiply(r1,np.multiply(np.cos(r2[r4>=0.5]),np.abs(np.multiply(r3[r4>=0.5],self.soluciones[self.idxMejorSolucion])-self.soluciones[r4>=0.5])))

    def getIndicadores(self):
        return self.indicadores
        