import inspect

class Metaheuristica:
    NUM_ITER = "num_iter"
    NP = "np"

    def __init__(self, ruta):
        print(f"metaheuristica abstracta creada")

    def setProblema(self, problema):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")
    
    def setParametros(self, parametros):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")

    def getParametros(self):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")

    def getIndicadores(self):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")
        
    def realizarBusqueda(self):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")

    def generarPoblacion(self, numero):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")
