

class MH_PRUEBA:
    def __init__(self):
        self.problema = None
        self.parametros = None
        print(f"Instancia de MH_PRUEBA creada")

    def setParametros(self,parametros):
        self.parametros = parametros
    
    def getParametros(self):
        return self.parametros

    def setProblema(self,problema):
        self.problema=problema
    
    def getProblema(self):
        return self.problema

    def getParamIniciales(self):
        return self.parametros

    def aplicarBusqueda(self):
        print(f"Aplicando busqueda")
        indicadores = Indicadores()
        return indicadores

class Indicadores:
    def __init__(self):
        print(f"Creando indicadores")