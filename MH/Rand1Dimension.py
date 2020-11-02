from MH.Metaheuristica import Metaheuristica
import numpy as np
from DTO.IndicadoresMH import IndicadoresMH
from DTO import TipoIndicadoresMH

class Rand1Dimension(Metaheuristica):
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
        print(f"Mh Rand1Dimension creada")
        

    
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
            ,TipoIndicadoresMH.FACTOR_EVOLUTIVO:self.getFactorEvolutivo()
        }

    def _perturbarSoluciones(self):
        idsPerturbar = np.random.randint(low=0,high=self.problema.getNumDim(), size=(self.soluciones.shape[0]))
        columnas = np.arange(self.soluciones.shape[0])
        random = np.random.uniform(low=-1,high=1,size=(self.soluciones.shape[0])) * self.parametros['salto']
        
        self.soluciones[columnas, idsPerturbar] += random.T

    def getIndicadores(self):
        return self.indicadores

    def getFactorEvolutivo(self):
        if self.idxMejorSolucion is None: return None
        if self.soluciones is None: return None
        dg = self.calcDistProm(self.idxMejorSolucion, self.soluciones)
        dmax = None
        dmin = None
        for i in range(self.soluciones.shape[0]):
            if i == self.idxMejorSolucion: continue
            dg = self.calcDistProm(i, self.soluciones)
            if dmax is None or dg > dmax: dmax = dg
            if dmin is None or dg < dmin: dmin = dg
        return (dg-dmin)/(dmax-dmin)

    def calcDistProm(self, idxSol, sols):
        res = 0
        for idx in range(len(sols)):
            if idx == idxSol: continue
            res += np.sum(np.abs(sols[idx]-sols[idxSol]))
        return res/len(sols)-1
        