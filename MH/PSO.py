from MH.Metaheuristica import Metaheuristica
import numpy as np
from DTO.IndicadoresMH import IndicadoresMH
from DTO import TipoIndicadoresMH

class PSO(Metaheuristica):
    C1 = "c1"
    C2 = "c2"
    W = "w"
    W_MIN = 0.2
    W_MAX = 0.99
    MAX_V = "max_vel"
    MIN_V = "min_vel"

    def __init__(self):
        self.problema = None
        self.soluciones = None
        self.parametros = {}
        self.idxMejorSolucion = None
        self.mejoraPorSol = None
        self.mejoraPorSolAcumulada = None
        self.mejorSolHistorica = None
        self.mejorFitHistorica = None
        self.fitness = None
        self.velocidad = None
        print(f"Mh PSO creada")
        

    
    def setProblema(self, problema):
        self.problema = problema

    def getProblema(self):
        return self.problema

    def generarPoblacion(self, numero):
        self.soluciones = self.problema.generarSoluciones(numero)
        self.mejorSolHistorica = self.problema.generarSoluciones(numero)
        self.fitness = self.problema.evalObj(self.soluciones)
        self.mejorFitHistorica = self.problema.evalObj(self.mejorSolHistorica)
        self.calcularEvaluaciones()
    
    def setParametros(self, parametros):
        for parametro in parametros:
            self.parametros[parametro] = parametros[parametro]
    
    def getParametros(self):
        return self.parametros

    def getFactorEvolutivo(self):
        if self.idxMejorSolucion is None: return None
        if self.soluciones is None: return None
        dg = self.calcDistProm(self.idxMejorSolucion, self.soluciones)
        dmax = None
        dmin = None
        #print("***")
        for i in range(self.soluciones.shape[0]):
            #if i == self.idxMejorSolucion: continue
            di = self.calcDistProm(i, self.soluciones)
            #print(di)
            if dmax is None or di > dmax: dmax = di
            if dmin is None or di < dmin: dmin = di
        #print("***")
        #print(f"{dg-dmin} {dmax-dmin}")
        if dg-dmin == 0: return 0
        if dmax-dmin == 0: return 1
        return (dg-dmin)/(dmax-dmin)

    def agregarParticula(self, numero):
        numero = int(numero)
        if numero > 0:
            soluciones = list(self.soluciones)
            mejorHistorico = list(self.mejorSolHistorica)
            velocidades = list(self.velocidad)
            fit = list(self.fitness)
            fitHist = list(self.mejorFitHistorica)
            mejoraPorSol = list(self.mejoraPorSol)
            mejoraPorSolAcumulada = list(self.mejoraPorSolAcumulada)
            nSol = self.problema.generarSoluciones(numero)
            nSolFit = self.problema.evalObj(nSol)
            mHist = self.problema.generarSoluciones(numero)
            mHistFit = self.problema.evalObj(mHist)
            idx = self.problema.getIndsMejora(mHistFit, nSolFit) > 0
            mHist[idx] = nSol[idx]
            mHistFit[idx] = nSolFit[idx]
            nSol = list(nSol)
            mHist = list(mHist)
            
            vel = np.random.uniform(low=self.parametros[PSO.MIN_V],high=self.parametros[PSO.MAX_V],size=(numero, self.soluciones.shape[1]))
            soluciones.extend(nSol)
            mejorHistorico.extend(mHist)
            velocidades.extend(vel)
            fit.extend(nSolFit)
            fitHist.extend(mHistFit)
            mejoraPorSol.extend(list(np.zeros((numero))))
            mejoraPorSolAcumulada.extend(list(np.zeros((numero))))
            self.soluciones = np.array(soluciones)
            self.mejorSolHistorica = np.array(mejorHistorico)
            self.velocidad = np.array(velocidades)
            self.fitness = np.array(fit)
            self.mejorFitHistorica = np.array(fitHist)
            self.mejoraPorSol = np.array(mejoraPorSol)
            self.mejoraPorSolAcumulada = np.array(mejoraPorSolAcumulada)
        else:
            for _ in range(-numero):
                peorIndice = self.problema.getPeorIdx(self.fitness)
                self.soluciones = np.delete(self.soluciones, peorIndice, 0)
                self.fitness = np.delete(self.fitness, peorIndice, 0)
                self.mejorSolHistorica = np.delete(self.mejorSolHistorica, peorIndice, 0)
                self.mejorFitHistorica = np.delete(self.mejorFitHistorica, peorIndice, 0)
                self.velocidad = np.delete(self.velocidad, peorIndice, 0)
                self.mejoraPorSol = np.delete(self.mejoraPorSol, peorIndice, 0)
                self.mejoraPorSolAcumulada = np.delete(self.mejoraPorSolAcumulada, peorIndice, 0)


    def calcDistProm(self, idxSol, sols):
        res = 0
        for idx in range(len(sols)):
            if idx == idxSol: continue
            res += np.sum(np.abs(sols[idx]-sols[idxSol]))
        return res/len(sols)

    def realizarBusqueda(self):
        if self.velocidad is None:
            self.velocidad = self.generarVelocidadInicial()
        self.agregarParticula(self.parametros[PSO.NP]-self.soluciones.shape[0])
        self._perturbarSoluciones()
        self.calcularEvaluaciones()
        #print(f"solucinoes {self.soluciones.shape}")
        
    def calcularEvaluaciones(self):
        fitness = self.problema.evaluarFitness(self.soluciones)
        assert self.soluciones.shape[0] == fitness.shape[0], "El numero de fitness es diferente al numero de soluciones"
        if self.fitness is None: self.fitness = fitness
        self.idxMejorSolucion = self.problema.getMejorIdx(fitness)
        self.mejoraPorSol = self.problema.getIndsMejora(self.fitness, fitness)
        assert self.soluciones.shape[0] == self.mejoraPorSol.shape[0], "El numero de indices de mejora es diferente al numero de soluciones"
        if self.mejoraPorSolAcumulada is None: self.mejoraPorSolAcumulada = np.zeros((self.soluciones.shape[0]))
        self.mejoraPorSolAcumulada += self.mejoraPorSol
        if self.mejorSolHistorica is None: self.mejorSolHistorica = self.soluciones
        if self.mejorFitHistorica is None: self.mejorFitHistorica = fitness

        mejorIdx = self.problema.getIndsMejora(self.mejorFitHistorica, fitness) > 0
        self.mejorSolHistorica[mejorIdx] = self.soluciones[mejorIdx]
        self.mejorFitHistorica[mejorIdx] = fitness[mejorIdx]


        self.fitness = fitness

        
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
        
        aceleracionPersonal = self.parametros[PSO.C1]*(self.mejorSolHistorica-self.soluciones)
        aceleracionGlobal = self.parametros[PSO.C2]*(self.mejorSolHistorica[self.problema.getMejorIdx(self.mejorFitHistorica)] - self.soluciones)
        inercia = self.parametros[PSO.W] * self.velocidad
        velocidad = (inercia + aceleracionPersonal + aceleracionGlobal)*self.parametros[PSO.MAX_V]
        #velocidad[velocidad>self.parametros[PSO.MAX_V]] = self.parametros[PSO.MAX_V]
        #velocidad[velocidad<self.parametros[PSO.MIN_V]] = self.parametros[PSO.MIN_V]
        self.soluciones += velocidad
        self.velocidad = velocidad
        self.soluciones[self.soluciones < self.problema.getDominioDim()[0]] = self.problema.getDominioDim()[0]
        self.soluciones[self.soluciones > self.problema.getDominioDim()[1]] = self.problema.getDominioDim()[1]        

    def generarVelocidadInicial(self):
        res = np.random.uniform(low=self.parametros[PSO.MIN_V],high=self.parametros[PSO.MAX_V],size=(self.soluciones.shape))
        #print("aqui")
        return res
        

    def getIndicadores(self):
        return self.indicadores
        