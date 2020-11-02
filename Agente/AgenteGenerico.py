from DTO import TipoIndicadoresMH, TipoParametro, TipoComponente, TipoIndicadoresMH, TipoDominio
import numpy as np

class AgenteGenerico:
    def __init__(self):
        self.parametrosAuto = None
        self.parametros = None
        self.indiceMejora = 0
        self.mejoraAcumulada = 0
        self.factorEvolutivo = []
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
        self.factorEvolutivo.append(indicadores[TipoIndicadoresMH.FACTOR_EVOLUTIVO])
              
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
        for parametro in self.parametrosAuto:
            if (self.mejoraAcumulada < 0.01 
                and parametro.getComponente() == TipoComponente.PROBLEMA):
                if parametro.getTipo() == TipoDominio.CONTINUO :
                    ret[parametro.nombre] = np.random.uniform(low=parametro.getMinimo(),high=parametro.getMaximo())
                if parametro.getTipo() == TipoDominio.DISCRETO :
                    ret[parametro.nombre] = np.random.randint(low=parametro.getMinimo(),high=parametro.getMaximo()+1)
        return ret

