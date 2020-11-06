from DTO.Resultado import Resultado
from DTO import TipoIndicadoresMH
from MH.Metaheuristica import Metaheuristica as MH
import json
from BD import RegistroMH
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


class GenericSolver:
    def __init__(self):
        print(f"Creando solver")
        self.mh = None
        self.agente = None
        self.numIterGuardar = 50

    def setMH(self,mh):
        self.mh = mh

    def getMH(self):
        return self.mh

    def setAgente(self, agente):
        self.agente = agente

    def getAgente(self):
        return self.agente

    def resolverProblema(self, idExperimento):
        print(f"Resolviendo problema")
        assert self.mh is not None, "No se ha iniciado la MH"
        assert self.agente is not None, "No se ha iniciado el Agente"

        self.mh.generarPoblacion(self.mh.getParametros()[MH.NP])
        resultadosIter = []
        fitness = []
        mejora = []
        facEvol = []
        
        
        #plt.ion()
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #sc = ax.scatter(self.mh.soluciones[:,0],self.mh.soluciones[:,1]) # Returns a tuple of line objects, thus the comma
        
        for i in range(self.mh.getParametros()[MH.NUM_ITER]):
            print(f'iter: {i}')
            inicio = datetime.now()
            self.mh.setIteracionActual(i)
            self.mh.realizarBusqueda()
            indicadores = self.mh.getIndicadores()
            self.agente.observarIndicadores(indicadores)
            paramOptimizadosMH = self.agente.optimizarParametrosMH(self.mh.getParametros())
            paramOptimizadosProblema = self.agente.optimizarParametrosProblema(self.problema.getParametros())
            self.mh.setParametros(paramOptimizadosMH)
            self.mh.problema.setParametros(paramOptimizadosProblema)
            fin = datetime.now()
            data = {
                "id_ejecucion": idExperimento
                ,"numero_iteracion": i
                ,"fitness_mejor": int(indicadores[TipoIndicadoresMH.FITNESS_MEJOR_GLOBAL])
                ,"fitness_promedio": float(indicadores[TipoIndicadoresMH.FITNESS_PROMEDIO])
                ,"fitness_mejor_iteracion": int(indicadores[TipoIndicadoresMH.FITNESS_MEJOR_ITERACION])
                ,"parametros_iteracion": json.dumps({"paramAgente": json.dumps(self.agente.getParametros())
                                            ,"paramMH": json.dumps(self.mh.getParametros())
                                            ,"paramProblema": json.dumps(self.mh.problema.getParametros())
                                            })
                ,"inicio": inicio
                ,"fin": fin
                ,"datos_internos": None
            }
            resultadosIter.append(data)
            if i % self.numIterGuardar == 0:
                RegistroMH.guardaDatosIteracion(resultadosIter)                
                resultadosIter = []
            print(f"Mejor fitness {'%.9f'%(self.mh.problema.getMejorEvaluacion())}\tmejora acumulada {'%.3f'%(self.agente.mejoraAcumulada)}")
            fitness.append(self.mh.problema.getMejorEvaluacion())
            mejora.append(self.agente.mejoraAcumulada)
            
        if len(resultadosIter) > 0:
            RegistroMH.guardaDatosIteracion(resultadosIter)
        #self.agente.guardarTablaQ()
        resultados = Resultado()
        resultados.setFitness(self.mh.problema.getMejorEvaluacion())
        resultados.setMejorSolucion(json.dumps(self.mh.soluciones[self.mh.idxMejorSolucion].tolist()))
        print(f"Problema resuelto, mejor fitness {self.mh.problema.getMejorEvaluacion()}")
        return resultados