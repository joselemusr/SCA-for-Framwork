from Problema.Problema import Problema
import numpy as np

class Esfera(Problema):
    def __init__(self):
        self.numDimensiones = 2
        minimo = np.ones(self.numDimensiones) * -100
        maximo = np.ones(self.numDimensiones) * 100
        #self.dominioDimensiones = np.array([minimo,maximo])
        self.dominioDimensiones = np.array([-100,100])
        self.mejorEvaluacion = None
        self.mejoresSoluciones = None
        self.parametros = {}
        print(f"Problema esfera creado")

    def getIndiceMejora(self):
        return self.indiceMejora
    
    def getMejorEvaluacion(self):
        return self.mejorEvaluacion

    def getMejoresSoluciones(self):
        return self.mejoresSoluciones

    def getNombre(self):
        return "Esfera"

    def setNumDim(self, num):
        self.numDimensiones = num

    def setParametros(self, parametros):
        for parametro in parametros:
            self.parametros[parametro] = parametros[parametro]

    def getParametros(self):
        return self.parametros
    
    def getNumDim(self):
        return self.numDimensiones

    def setDominioDim(self, dominio):
        self.dominioDimensiones = dominio
    
    def getDominioDim(self):
        return self.dominioDimensiones

    def generarSoluciones(self, numero):
        res = []
        if numero > 0:
            #minimos = self.getDominioDim()[0]
            minimos = np.array([-100,-100])
            #print(minimos)
            #exit()
            #maximos = self.getDominioDim()[1]
            maximos = np.array([100,100])
            if self.mejoresSoluciones is not None:
                promedio = np.average(self.mejoresSoluciones,axis=0)
                minimos = promedio - (promedio*0.1)
                maximos = promedio + (promedio*0.1)
            for i in range(self.getNumDim()):
                #print(f"minimo {i} {minimos}")
                #print(f"{minimos[i]}")
                res.append(np.random.uniform(low=minimos[i], high=maximos[i], size=(numero)))
                #res.append(np.ones((numero))*100)
                
        return np.array(res).T

    def decode(self, soluciones):
        return soluciones

    def evalObj(self, soluciones):
        return np.sum(np.square(soluciones),axis = 1)

    def evaluarFitness(self, soluciones):
        evaluaciones = self.evalObj(soluciones)
        mejorEvaluacion = np.min(evaluaciones)
        if self.mejorEvaluacion is None: self.mejorEvaluacion = mejorEvaluacion
        idxMejorEval = evaluaciones == mejorEvaluacion
        mejoresSoluciones = np.unique(soluciones[idxMejorEval], axis=0)
        self.indiceMejora = self.getIndsMejora(self.mejorEvaluacion,mejorEvaluacion)
        #print(f"indice mejora {self.indiceMejora}")
        if mejorEvaluacion < self.mejorEvaluacion:
            self.mejoresSoluciones = mejoresSoluciones
            self.mejorEvaluacion = mejorEvaluacion
        if mejorEvaluacion == self.mejorEvaluacion:
            mejoresSolucionesL = list(mejoresSoluciones)
            if self.mejoresSoluciones is not None:
                mejoresSolucionesL.extend(list(self.mejoresSoluciones))
            self.mejoresSoluciones = np.unique(np.array(mejoresSolucionesL), axis=0)
        
        #print(f"self.indiceMejora {self.indiceMejora} ({self.mejorEvaluacion}-{mejorEvaluacion})/{self.mejorEvaluacion} {(self.mejorEvaluacion-mejorEvaluacion)/self.mejorEvaluacion}")
        return evaluaciones

    def getIndsMejora(self, f1, f2):
        #cuanto mejora f2 a f1 
        assert f1.shape == f2.shape, f"Fitness 1 {f1.shape} diferente a fitness 2 {f2.shape}"
        return (f1-f2)/f1

    def getMejorIdx(self, fitness):
        return np.argmin(fitness)

    def getPeorIdx(self, fitness):
        return np.argmax(fitness)