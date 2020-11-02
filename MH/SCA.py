from MH.Metaheuristica import Metaheuristica
import numpy as np
from DTO.IndicadoresMH import IndicadoresMH
from DTO import TipoIndicadoresMH
import math

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
    
    def setParametros(self, parametros):
        for parametro in parametros:
            self.parametros[parametro] = parametros[parametro]
    
    def getParametros(self):
        return self.parametros

    def realizarBusqueda(self):
        self._perturbarSoluciones()
        fitness = self.problema.evaluarFitness(self.problema.decode(self.soluciones))
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

        
        #minFitness = np.min(fitness)
        #indicadorFitness = IndicadoresMH()
        #indicadorFitness.setNombre(TipoIndicadoresMH.FITNESS)
        #indicadorFitness.setValor(minFitness)
        #indicadorMejora = IndicadoresMH()
        #indicadorMejora.setNombre(TipoIndicadoresMH.INDICE_MEJORA)
        #indicadorMejora.setValor(self.problema.getIndiceMejora())
        #self.indicadores = [indicadorFitness, indicadorMejora]
        self.indicadores = {
            TipoIndicadoresMH.INDICE_MEJORA:self.problema.getIndiceMejora()
            ,TipoIndicadoresMH.FITNESS_MEJOR_GLOBAL:self.problema.getMejorEvaluacion()
            ,TipoIndicadoresMH.FITNESS_MEJOR_ITERACION:fitness[self.idxMejorSolucion]
            ,TipoIndicadoresMH.FITNESS_PROMEDIO:np.mean(fitness)
        }

    def _perturbarSoluciones(self):        
        for i in range(self.problema.getNumDim()):

            if self.idxMejorSolucion is None: #Para la primera iteración
                self.idxMejorSolucion = np.random.randint(low=0,high=self.soluciones.shape[0])

            #idsPerturbar = np.random.randint(low=0,high=self.problema.getNumDim(), size=(self.soluciones.shape[0]))
            #columnas = np.arange(self.soluciones.shape[0])
                
            r1 = self.parametros['a'] - (self.IteracionActual*(self.parametros['a']/self.getParametros()["numIter"])) #Escalar
            r4 = np.random.uniform(low=0.0,high=1.0)
            
            if r4 < 0.5: #El paper define el criterio de elección como 0.5 "valor duro"

                r2 =(2*np.pi) * np.random.uniform(low=0.0,high=1.0, size=(self.soluciones.shape[0]))
                r3 = np.random.uniform(low=0.0,high=2.0, size=(self.soluciones.shape[0]))

                movimiento = (r1 * np.sin(r2)) * np.absolute((r3 * self.soluciones[self.idxMejorSolucion][i]) - self.soluciones[:, i]).T

                self.soluciones[:, i] += movimiento.T

            else:

                r2 =(2*np.pi) * np.random.uniform(low=0.0,high=1.0, size=(self.soluciones.shape[0]))
                r3 = np.random.uniform(low=0.0,high=2.0, size=(self.soluciones.shape[0]))

                movimiento = (r1 * np.cos(r2)) * np.absolute((r3 * self.soluciones[self.idxMejorSolucion][i]) - self.soluciones[:, i]).T

                self.soluciones[:, i] += movimiento.T

    def getIndicadores(self):
        return self.indicadores
        