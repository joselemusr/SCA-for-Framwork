from DTO import TipoIndicadoresMH, TipoParametro, TipoComponente, TipoIndicadoresMH, TipoDominio
from MH.Metaheuristica import Metaheuristica as MH
from hmmlearn import hmm
import numpy as np
from MH.PSO import PSO
import math
import pickle
import os

class AgenteQL_HMM:

    def decrease(self, param, tipo=TipoDominio.CONTINUO, minimo=None, maximo=None):
        if tipo == TipoDominio.DISCRETO : 
            param -= 2
        else:
            param *= np.random.uniform(low=0.9,high=0.95)
        if tipo == TipoDominio.DISCRETO : 
            param = int(param)
        if maximo is not None and param > maximo:
            param = maximo
        if minimo is not None and param < minimo:
            param = minimo
        return param

            
    def lDecrease(self, param, tipo=TipoDominio.CONTINUO, minimo=None, maximo=None):
        if tipo == TipoDominio.DISCRETO : 
            param -= 1
        else:
            param *= np.random.uniform(low=0.95,high=0.99)
        if tipo == TipoDominio.DISCRETO : 
            param = int(param)
        if maximo is not None and param > maximo:
            param = maximo
        if minimo is not None and param < minimo:
            param = minimo
        return param

            
    def none(self, param, tipo=TipoDominio.CONTINUO, minimo=None, maximo=None):
        return param

            
    def lIncrease(self, param, tipo=TipoDominio.CONTINUO, minimo=None, maximo=None):
        if tipo == TipoDominio.DISCRETO : 
            param += 1
        else:
            param *= np.random.uniform(low=1.01,high=1.05)
        if tipo == TipoDominio.DISCRETO : 
            param = int(param)
        if maximo is not None and param > maximo:
            param = maximo
        if minimo is not None and param < minimo:
            param = minimo
        return param

            
    def increase(self, param, tipo=TipoDominio.CONTINUO, minimo=None, maximo=None):
        
        if tipo == TipoDominio.DISCRETO : 
            param += 2
        else:
            param *= np.random.uniform(low=1.05,high=1.1)
        if maximo is not None and param > maximo:
            param = maximo
        if minimo is not None and param < minimo:
            param = minimo
        return param

            

    def __init__(self):
        self.FEDiscreto = []
        self.parametrosAuto = None
        self.parametros = None
        self.indiceMejora = 0
        self.mejoraAcumulada = 0
        self.factorEvolutivo = []
        self.estado = 0
        self.estadoAnterior = 0
        self.states = ['Exploration', 'Exploitation', 'Convergence', 'Jump out']
        self.actions = [self.decrease, self.lDecrease, self.none, self.lIncrease, self.increase]
        self.gamma = 1        
        self.lrate = 1
        self.minLRate = 0.003
        self.qtable = None
        self.accion = None
        self.accionAnterior = None
        self.totIter = 0
        self.numIter = 0
        self.eps = 0.03
        self.partIter = 10
        
        self.initHMM()
        print(f"Instancia de agente hmm creada")

    def initHMM(self):
        
        if os.path.exists('hmm-model.pkl'):
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
            
            self.dhmm = hmm.MultinomialHMM(n_components=n_states, n_iter=100, verbose=True, init_params="s", params='e')
            #self.dhmm = hmm.MultinomialHMM(n_components=n_states, startprob_prior=Pi, transmat_prior=A, init_params="",  n_iter=100)
            self.dhmm.n_features = 7
            #self.dhmm.n_features_ = 7
            hmm.normalize(B,axis=1)
            self.dhmm.transmat_=A
            self.dhmm.transmat_prior=A
            self.dhmm.emissionprob_=B
            self.dhmm.emissionprob_prior=B
            self.dhmm.startprob_=Pi
            self.dhmm.monitor_.verbose = True
            self.dhmm.monitor_.tol = 0.0001
            #print(self.dhmm.monitor_)
            #exit()
            #self.dhmm.startprob_prior = Pi
            #with open("hmm-model.pkl", "wb") as file: pickle.dump(self.dhmm, file)
        

    def setParametrosAutonomos(self, parametros):
        self.parametrosAuto = parametros
        self.accion = np.ones((len(parametros)), dtype=np.int8) * 2
        self.accionAnterior = np.ones((len(parametros)), dtype=np.int8) * 2
        if os.path.exists('qtable.npy'):
            with open('qtable.npy', "rb") as file: 
                self.qtable = np.load(file)
        else:
            self.qtable = np.zeros((len(parametros),self.partIter+1, len(self.states),len(self.actions)))
        
    def setTotIter(self, totIter):
        self.totIter = totIter

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
        print(f"np.unique(data).shape[0] {np.unique(data).shape[0]} self.dhmm.n_features {self.dhmm.n_features}")
        if np.unique(data).shape[0] >= self.dhmm.n_features:
            #print(f"entrenando hmm")
            #print(self.dhmm.emissionprob_)
            #print(self.dhmm.startprob_)
            #for _ in range(10):
            self.dhmm.fit(data)
            #score = self.dhmm.score(data)
            #print(f"score de hmm : {score}")
            
            with open("hmm-model.pkl", "wb") as file: pickle.dump(self.dhmm, file)

        ( log_prob, estado ) = self.dhmm.decode(data, algorithm="viterbi")
        #estado = self.dhmm.predict(data)
        print(f"log_prob {np.exp(log_prob)} converged {self.dhmm.monitor_.converged}")
        #print(estado)
        #print(np.array(self.FEDiscreto))
        #print(f"self.dhmm.startprob_ {self.dhmm.startprob_}")
        #print(f"self.dhmm.transmat_ {self.dhmm.transmat_}")
        #print(f"self.dhmm.emissionprob_ {self.dhmm.emissionprob_}")
        self.estadoAnterior = self.estado
        self.estado = estado[-1]
        #actualizar tabla q para cada parametro
        grupoAnt = int(self.numIter / (self.totIter/self.partIter))
        self.numIter += 1
        grupo = int(self.numIter / (self.totIter/self.partIter))
        eta = max(self.minLRate, self.lrate * (0.85 ** (self.numIter//100)))
        for i in range(len(self.parametrosAuto)):
            #print(f"self.qtable[i] {self.qtable[i]}")
            #print(f"self.accion {self.accion}")
            a = self.gamma * np.max(self.qtable[i][grupo][self.estado][self.accion[i]])
            #b = self.indiceMejora + a
            b = self.mejoraAcumulada + a
            c = eta * b
            d = c - self.qtable[i][grupoAnt][self.estadoAnterior][self.accionAnterior[i]]
            self.qtable[i][grupoAnt][self.estadoAnterior][self.accionAnterior] += d
        
        
        
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
        self.accionAnterior = self.accion
        i = 0
        for parametro in self.parametrosAuto:
            if np.random.uniform(0, 1) < self.eps:
                self.accion[i] = np.random.choice(len(self.actions))
            else:
                #logits = self.qtable[i][self.estado]
                #logits_exp = np.exp(logits)
                #probs = np.nan_to_num(logits_exp / np.sum(logits_exp))
                #self.accion[i] = np.random.choice
                # (len(self.actions), p=probs)
                grupo = int(self.numIter / (self.totIter/self.partIter))
                self.accion[i] = np.argmax(self.qtable[i][grupo][self.estado])
            
            ret[parametro.getNombre()] = self.actions[self.accion[i]](params[parametro.getNombre()], parametro.getTipo(), parametro.getMinimo(), parametro.getMaximo())
            i += 1
            
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

    def guardarTablaQ(self):
        np.save('qtable.npy', self.qtable)
        
