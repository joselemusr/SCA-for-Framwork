from DTO import TipoIndicadoresMH, TipoParametro, TipoComponente, TipoIndicadoresMH, TipoDominio
import numpy as np

class Qlearning():
    GAMMA = "Gamma"
    ACTIONS = "Actions"
    STATETYPE = "stateType"
    QLALPHATYPE = "qlAlphaType"
    REWARDTYPE = "rewardType"
    POLICYTYPE = "PolicyType"
    ITERMAX = "iterMax"
    EPSILON = "epsilon"
    QLALPHA = "qlAlpha"

    def __init__(self):
        self.parametrosAuto = None
        self.parametros = None
        self.indiceMejora = 0
        self.mejoraAcumulada = 0
        
        self.gamma = gamma
        self.qlAlphaType = qlAlphaType
        self.rewardType = rewardType
        self.iterMax = iterMax
        
        
        self.epsilon = epsilon
        self.qlAlpha = qlAlpha
        self.bestMetric = 999999999 #Esto en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
        self.Qvalues = np.zeros(shape=(self.parametros[Qlearning.STATETYPE],self.parametros[Qlearning.ACTIONS])) #state,actions
        self.visitas = np.zeros(shape=(self.parametros[Qlearning.STATETYPE],self.parametros[Qlearning.ACTIONS])) #state,actions
        #self.Qvalues[0] = np.zeros(len(self.actions))

        print(f"Instancia de agente generico creada")

    def setTotIter(self, t):
        self.totIter = t

    def setParametrosAutonomos(self, parametros):
        self.parametrosAuto = parametros

    def getParametrosAutonomos(self):
        return self.parametrosAuto

    def setParametros(self, parametros):
        self.parametros = parametros

    def getParametros(self):
        return self.parametros
    
    def observarIndicadores(self,indicadores):
        self.indiceMejora = indicadores[TipoIndicadoresMH.INDICE_MEJORA]
        self.mejoraAcumulada += indicadores[TipoIndicadoresMH.INDICE_MEJORA]
              
    def optimizarParametrosMH(self, params):
        ret = {}
        for parametro in self.parametrosAuto:
            if (self.mejoraAcumulada < 0 
                and parametro.getComponente() == TipoComponente.METAHEURISTICA):
                if parametro.getTipo() == TipoDominio.CONTINUO :
                    ret[parametro.getNombre()] = np.random.uniform(low=parametro.getMinimo(),high=parametro.getMaximo())
                if parametro.getTipo() == TipoDominio.DISCRETO :
                    ret[parametro.getNombre()] = np.random.randint(low=parametro.getMinimo(),high=parametro.getMaximo()+1)
        return ret

    def optimizarParametrosProblema(self):
        ret = {}
        BinariyScheme  = {}
        #Leer Indicadores
        state = "Leer de Indicador"

        NumAction = self.getAccion(state)    

        BinariyScheme[tBinary] = "V4"
        BinariyScheme[tTransferencia] = "Standar"
        for parametro in self.parametrosAuto:
            ret[parametro.nombre] = BinariyScheme[parametro]
        return ret



    def getReward(self,metric):
        
        if self.rewardType == "withPenalty1": 
            if self.bestMetric > metric:#Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
                self.bestMetric = metric
                return 1
            return -1

        elif self.rewardType == "withoutPenalty1":
            if self.bestMetric > metric:#Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
                self.bestMetric = metric
                return 1
            return 0

    def getAccion(self,state):
        
        # e-greedy
        if self.parametros[Qlearning.POLICYTYPE] == "e-greedy":
            probabilidad = np.random.uniform(low=0.0, high=1.0) #numero aleatorio [0,1]
            if probabilidad <= self.epsilon: #seleccion aleatorio
                return np.random.randint(low=0, high=self.Qvalues.shape[0]) #seleccion aleatoria de una accion     
            else: #selecion de Q_Value mayor        
                maximo = np.amax(self.Qvalues,axis=1) # retorna el elemento mayor por fila        
                indices = np.where(self.Qvalues[state,:] == maximo[state])[0]  #retorna los indices donde se ubica el maximo en la fila estado        
                return np.random.choice(indices) # funciona tanto cuando hay varios iguales como cuando hay solo uno 
        
        # greedy
        elif self.parametros[Qlearning.POLICYTYPE] == "greedy":
            return np.argmax(self.Qvalues[state])

        # e-soft 
        elif self.parametros[Qlearning.POLICYTYPE] == "e-soft":
            probabilidad = np.random.uniform(low=0.0, high=1.0) #numero aleatorio [0,1]
            if probabilidad > self.epsilon: #seleccion aleatorio
                return np.random.randint(low=0, high=self.Qvalues.shape[0]) #seleccion aleatoria de una accion     
            else: #selecion de Q_Value mayor        
                maximo = np.amax(self.Qvalues,axis=1) # retorna el elemento mayor por fila        
                indices = np.where(self.Qvalues[state,:] == maximo[state])[0]  #retorna los indices donde se ubica el maximo en la fila estado        
                return np.random.choice(indices) # funciona tanto cuando hay varios iguales como cuando hay solo uno 

        # softMax seleccion ruleta
        elif self.parametros[Qlearning.POLICYTYPE] == "softMax-rulette":
            #*** Falta generar una normalización de las probabilidades que sumen 1, para realizar el choice
            Qtable_normalizada = np.nan_to_num(self.Qvalues[state] / np.linalg.norm(self.Qvalues[state])) # normalizacion de valores   
            #La suma de las prob = 1
            seleccionado = np.random.choice(self.Qvalues[state],p=Qtable_normalizada)
            indices = np.where(self.Qvalues[state,:] == seleccionado)[0]
            return np.random.choice(indices)
    
        # softmax seleccion ruleta elitista (25% mejores acciones)
        elif self.parametros[Qlearning.POLICYTYPE] == "softMax-rulette-elitist":
            sort = np.argsort(self.Qvalues[state]) # argumentos ordenados
            cant_mejores = int(sort.shape[0]*0.25) # obtenemos el 25% de los mejores argumentos
            rulette_elitist = sort[0:cant_mejores] # tiene el 25% de los mejores argumentos
            return np.random.choice(rulette_elitist)
    


    def actualizar_Visitas(self,action,state): # ACTUALIZACION DE LAS VISITAS
        self.visitas[state,action] = self.visitas[state,action] + 1


    def getAlpha(self,state,action,iter):

        if self.qlAlphaType == "static": 
            #alpha estatico 
            return  self.qlAlpha

        elif self.qlAlphaType == "iteration":
            return 1 - (0.9*(iter/self.iterMax))
            
        elif self.qlAlphaType == "visits":
            return (1/(1 + self.visitas[state,action]))

    def updateQtable(self,metric,action,state,oldState,iter):

        #revisar

        Reward = self.getReward(metric)
        
        alpha = self.getAlpha(oldState,action,iter)

        Qnuevo = ( (1 - alpha) * self.Qvalues[oldState][action]) + alpha * (Reward + (self.gamma  * max(self.Qvalues[state])))

        self.actualizar_Visitas(action,oldState) #Actuzación de visitas

        #REVISAR ESTO
        self.Qvalues[state][action] = Qnuevo

    def getQtable(self):
        return self.Qvalues