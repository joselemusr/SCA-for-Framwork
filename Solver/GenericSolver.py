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
        self.numIterGuardar = 20

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
            inicio = datetime.now()
            self.mh.realizarBusqueda()
            indicadores = self.mh.getIndicadores()
            self.agente.observarIndicadores(indicadores)
            paramOptimizadosMH = self.agente.optimizarParametrosMH(self.mh.getParametros())
            paramOptimizadosProblema = self.agente.optimizarParametrosProblema()
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
            print(f"Mejor fitness {'%.9f'%(self.mh.problema.getMejorEvaluacion())}\tmejora acumulada {'%.3f'%(self.agente.mejoraAcumulada)}\tfactor evolutivo {'%.3f'%(indicadores[TipoIndicadoresMH.FACTOR_EVOLUTIVO])} \thmm {self.agente.states[self.agente.estado]}")
            fitness.append(self.mh.problema.getMejorEvaluacion())
            mejora.append(self.agente.mejoraAcumulada)
            facEvol.append(indicadores[TipoIndicadoresMH.FACTOR_EVOLUTIVO])
            
            #X = list(self.mh.soluciones)
            #X.append(np.ones((self.mh.soluciones.shape[1]))*self.mh.problema.getDominioDim()[1])
            #X.append(np.ones((self.mh.soluciones.shape[1]))*self.mh.problema.getDominioDim()[0])
            #X = np.array(X)
            #tsne = TSNE(n_components=2, init="pca")
            #X_embedded = tsne.fit_transform(X)
            #sc.set_offsets(X_embedded)
            #plt.xlim(np.min(X_embedded[:,0])-100,np.max(X_embedded[:,0])+100)
            #plt.ylim(np.min(X_embedded[:,1])-100,np.max(X_embedded[:,1])+100)
            #ax.set_title(f'estado calculado {self.agente.states[self.agente.estado]}')
            #fig.canvas.draw()
            #fig.canvas.flush_events()
            
        if len(resultadosIter) > 0:
            RegistroMH.guardaDatosIteracion(resultadosIter)
        self.agente.guardarTablaQ()
        resultados = Resultado()
        resultados.setFitness(self.mh.problema.getMejorEvaluacion())
        resultados.setMejorSolucion(json.dumps(self.mh.soluciones[self.mh.idxMejorSolucion].tolist()))
        print(f"Problema resuelto, mejor fitness {self.mh.problema.getMejorEvaluacion()}")

        #fig, axs = plt.subplots(3)
        #axs[0].plot(fitness)
        #axs[1].plot(mejora)
        #axs[2].plot(facEvol)
        #axs[0].set_title('Fitness')
        #axs[1].set_title('Mejora Acumulada')
        #axs[2].set_title('Factor Evolutivo')
        #plt.show()
        return resultados