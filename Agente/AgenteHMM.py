from DTO import TipoIndicadoresMH, TipoParametro, TipoComponente, TipoIndicadoresMH, TipoDominio
from hmmlearn import hmm
import numpy as np
from MH.PSO import PSO
import math
import pickle
import os

class AgenteHMM:
    def __init__(self):
        self.FEDiscreto = []
        self.parametrosAuto = None
        self.parametros = None
        self.indiceMejora = 0
        self.mejoraAcumulada = 0
        self.factorEvolutivo = []
        self.estado = 0
        self.states = ['Exploration', 'Exploitation', 'Convergence', 'Jump out']
        if os.path.exists('hmm-model.pkl') and False:
            with open("hmm-model.pkl", "rb") as file: 
                self.dhmm = pickle.load(file)
        else:
            Pi = np.array([1,0,0,0])
            A = np.array([
                [0.5    ,0.5    ,0      ,0 ],
                [0      ,0.5    ,0.5    ,0 ],
                [0      ,0      ,0.5    ,0.5],
                [0.5    ,0      ,0      ,0.5]
            ])
            #B = np.array([
            #    [0.001      ,0.001      ,0.001      ,0.5-0.001    ,0.25-0.001   ,0.25-0.002   ,0.001  ],
            #    [0.001      ,0.25-0.001   ,0.25-0.001   ,0.5-0.002    ,0.001      ,0.001      ,0.001  ],
            #    [2/3-0.003    ,1/3-0.002    ,0.001      ,0.001      ,0.001      ,0.001      ,0.001  ],
            #    [0.001      ,0.001      ,0.001      ,0.001      ,0.001      ,1/3-0.002    ,2/3-0.003]
            #])
            B = np.array([
                [0      ,0      ,0      ,0.5    ,0.25   ,0.25   ,0  ],
                [0      ,0.25   ,0.25   ,0.5    ,0      ,0      ,0  ],
                [2/3    ,1/3    ,0      ,0      ,0      ,0      ,0  ],
                [0      ,0      ,0      ,0      ,0      ,1/3    ,2/3]
            ])
            n_states = len(self.states)
            
            self.dhmm = hmm.MultinomialHMM(n_components=n_states, n_iter=100, verbose=False, init_params="s", params='es')
            #self.dhmm = hmm.MultinomialHMM(n_components=n_states, startprob_prior=Pi, transmat_prior=A, init_params="",  n_iter=100)
            self.dhmm.n_features = 7
            #self.dhmm.n_features_ = 7
            hmm.normalize(B,axis=1)
            self.dhmm.transmat_=A
            self.dhmm.transmat_prior=A
            self.dhmm.emissionprob_=B
            self.dhmm.emissionprob_prior=B
            self.dhmm.startprob_=Pi
            #self.dhmm.startprob_prior = Pi
            #with open("hmm-model.pkl", "wb") as file: pickle.dump(self.dhmm, file)
        print(f"Instancia de agente hmm creada")

    def setParametrosAutonomos(self, parametros):
        self.parametrosAuto = parametros

    def getParametrosAutonomos(self):
        return self.parametrosAuto

    def setParametros(self, parametros):
        self.parametros = parametros

    def getParametros(self):
        return self.parametros
    
    def observarIndicadores(self,indicadores):
        params =self.dhmm._get_n_fit_scalars_per_param()
        numParams = sum([params[nombre] for nombre in params])
        self.indiceMejora = indicadores[TipoIndicadoresMH.INDICE_MEJORA]
        self.mejoraAcumulada += indicadores[TipoIndicadoresMH.INDICE_MEJORA]
        self.factorEvolutivo.append(indicadores[TipoIndicadoresMH.FACTOR_EVOLUTIVO])
        self.FEDiscreto.append(self.disEstEvol(indicadores[TipoIndicadoresMH.FACTOR_EVOLUTIVO]))
        
        data = np.array([self.FEDiscreto]).T
        #print(data)
        estado = [0]
        
        if np.prod(data.shape) >= 80 and np.unique(data).shape[0] == self.dhmm.n_features:
            #print(f"entrenando hmm")
            #print(self.dhmm.emissionprob_)
            #print(self.dhmm.startprob_)
            self.dhmm.fit(data)
            
            with open("hmm-model.pkl", "wb") as file: pickle.dump(self.dhmm, file)

        ( log_prob, estado ) = self.dhmm.decode(data, algorithm="viterbi")
        #estado = self.dhmm.predict(data)
        #print(log_prob)
        #print(estado)
        #print(np.array(self.FEDiscreto))
        #print(f"self.dhmm.startprob_ {self.dhmm.startprob_}")
        #print(f"self.dhmm.transmat_ {self.dhmm.transmat_}")
        #print(f"self.dhmm.emissionprob_ {self.dhmm.emissionprob_}")
        self.estado = estado[-1]
        
    def disEstEvol(self, estado):
        if estado < 0.2: return 0
        if 0.2 <= estado < 0.3: return 1
        if 0.3 <= estado < 0.4: return 2
        if 0.4 <= estado < 0.6: return 3
        if 0.6 <= estado < 0.7: return 4
        if 0.7 <= estado < 0.8: return 5
        if 0.8 <= estado <= 1: return 6

    def optimizarParametrosMH(self, params):
        ret = {}
        
        for parametro in self.parametrosAuto:
            
            
            
            if self.states[self.estado] == 'Exploration':
#                if 'accelPer' in self.paramOptimizar:
#                    self.parametros['accelPer'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=self.aumentoRango[0],high=self.aumentoRango[1])
#                if 'accelBest' in self.paramOptimizar:
#                    self.parametros['accelBest'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=self.disminucionRango[0],high=self.disminucionRango[1])
                if parametro.getNombre() == PSO.C1:
                    if params[parametro.getNombre()] <= parametro.getMaximo():
                        ret[parametro.getNombre()] = params[PSO.C1] * np.random.uniform(low=1.05,high=1.1)
                if parametro.getNombre() == PSO.C2:
                    if params[parametro.getNombre()] >= parametro.getMinimo():
                        ret[parametro.getNombre()] = params[PSO.C1] * np.random.uniform(low=0.9,high=0.95)
                if parametro.getNombre() == PSO.NP:
                    if params[PSO.NP]+1 <= parametro.getMaximo():
                        ret[parametro.getNombre()] = params[PSO.NP] + 1
                    else: ret[parametro.getNombre()] = parametro.getMinimo()
                if parametro.getNombre() == PSO.W:
                    ret[parametro.getNombre()] = PSO.W_MIN + (PSO.W_MAX - PSO.W_MIN) * np.random.uniform()
                if parametro.getNombre() == PSO.MAX_V:
                    ret[parametro.getNombre()] = params[PSO.MAX_V]*1.001
            if self.states[self.estado] == 'Exploitation':
#                if 'accelPer' in self.paramOptimizar:
#                    self.parametros['accelPer'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=self.aumentoLeveRango[0],high=self.aumentoLeveRango[1])
#                if 'accelBest' in self.paramOptimizar:
#                    self.parametros['accelBest'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=self.disminucionLeveRango[0],high=self.disminucionLeveRango[1])
#                if 'numParticulas' in self.paramOptimizar:
#                    if self.parametros['nivel'] == 1:
#                        self.parametros['solPorGrupo'][idGrupo] -= 1
                if parametro.getNombre() == PSO.C1:
                    if params[parametro.getNombre()] <= parametro.getMaximo():
                        ret[parametro.getNombre()] = params[PSO.C1] * np.random.uniform(low=1.01,high=1.05)
                if parametro.getNombre() == PSO.C2:
                    if params[parametro.getNombre()] >= parametro.getMinimo():
                        ret[parametro.getNombre()] = params[PSO.C1] * np.random.uniform(low=0.95,high=0.99)
                if parametro.getNombre() == PSO.NP:
                    if params[PSO.NP]-1 >= parametro.getMinimo():
                        ret[parametro.getNombre()] = params[PSO.NP]-1
                if parametro.getNombre() == PSO.W:
                    ret[parametro.getNombre()] = 1/1+(1.5*math.exp(-2.6*self.factorEvolutivo[-1]))
                if parametro.getNombre() == PSO.MAX_V:
                    ret[parametro.getNombre()] = params[PSO.MAX_V]*0.999
            if self.states[self.estado] == 'Convergence':
#                if 'accelPer' in self.paramOptimizar:
#                    self.parametros['accelPer'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=self.aumentoLeveRango[0],high=self.aumentoLeveRango[1])
#                if 'accelBest' in self.paramOptimizar:
#                    self.parametros['accelBest'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=self.aumentoLeveRango[0],high=self.aumentoLeveRango[1])
#                if 'numParticulas' in self.paramOptimizar:
#                    if self.parametros['nivel'] == 1:
#                        self.parametros['solPorGrupo'][idGrupo] -= 1
                if parametro.getNombre() == PSO.C1:
                    if params[parametro.getNombre()] <= parametro.getMaximo():
                        ret[parametro.getNombre()] = params[PSO.C1] * np.random.uniform(low=1.01,high=1.05)
                if parametro.getNombre() == PSO.C2:
                    if params[parametro.getNombre()] <= parametro.getMaximo():
                        ret[parametro.getNombre()] = params[PSO.C1] * np.random.uniform(low=1.01,high=1.05)
                if parametro.getNombre() == PSO.NP:
                    if params[PSO.NP]+1 <= parametro.getMaximo():
                        ret[parametro.getNombre()] = params[PSO.NP]+1
                    else:
                        ret[parametro.getNombre()] = parametro.getMinimo()

                if parametro.getNombre() == PSO.W:
                    ret[parametro.getNombre()] = PSO.W_MIN
                #if parametro.getNombre() == PSO.MAX_V:
                #    ret[parametro.getNombre()] = params[PSO.MAX_V]*0.99
            if self.states[self.estado] == 'Jump out':
#                if 'numParticulas' in self.paramOptimizar:
#                    if self.parametros['nivel'] == 1:
#                        self.parametros['solPorGrupo'][idGrupo] += 1
#                if 'accelPer' in self.paramOptimizar:
#                    self.parametros['accelPer'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=self.disminucionRango[0],high=self.disminucionRango[1])
#                if 'accelBest' in self.paramOptimizar:
#                    self.parametros['accelBest'][self.parametros['nivel']][idGrupo] *= np.random.uniform(low=self.aumentoRango[0],high=self.aumentoRango[1])
                if parametro.getNombre() == PSO.C1:
                    if params[parametro.getNombre()] >= parametro.getMaximo():
                        ret[parametro.getNombre()] = params[PSO.C1] * np.random.uniform(low=0.9,high=0.95)
                if parametro.getNombre() == PSO.C2:
                    if params[parametro.getNombre()] <= parametro.getMaximo():
                        ret[parametro.getNombre()] = params[PSO.C1] * np.random.uniform(low=1.05,high=1.1)
                if parametro.getNombre() == PSO.NP:
                    if params[PSO.NP]-1 >= parametro.getMinimo():
                        ret[parametro.getNombre()] = params[PSO.NP]-1
                if parametro.getNombre() == PSO.W:
                    ret[parametro.getNombre()] = PSO.W_MAX
                if parametro.getNombre() == PSO.MAX_V:
                    ret[parametro.getNombre()] = params[PSO.MAX_V]*1.1
            

            
            
            #if parametro.getNombre() == PSO.W:
            #    ret[parametro.getNombre()] = 1/1+(1.5*math.exp(-2.6*self.factorEvolutivo[-1]))
            #if (self.mejoraAcumulada < 0 
            #    and parametro.getComponente() == TipoComponente.METAHEURISTICA):
            #    
            #    if parametro.getTipo() == TipoDominio.CONTINUO :
            #        ret[parametro.getNombre()] = np.random.uniform(low=parametro.getMinimo(),high=parametro.getMaximo())
            #    if parametro.getTipo() == TipoDominio.DISCRETO :
            #        ret[parametro.getNombre()] = np.random.randint(low=parametro.getMinimo(),high=parametro.getMaximo()+1)
        
        return ret

    def optimizarParametrosProblema(self):
        ret = {}
        for parametro in self.parametrosAuto:
            if (self.mejoraAcumulada < 0.01 
                and parametro.getComponente() == TipoComponente.PROBLEMA):
                if parametro.getTipo() == TipoDominio.CONTINUO :
                    ret[parametro.nombre] = np.random.uniform(low=parametro.getMinimo(),high=parametro.getMaximo())
                if parametro.getTipo() == TipoDominio.DISCRETO :
                    ret[parametro.nombre] = np.random.randint(low=parametro.getMinimo(),high=parametro.getMaximo()+1)
        return ret

