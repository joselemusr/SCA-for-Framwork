import inspect

class Problema:
    TRANSFER_FUNCTION = "Function_Transfer"
    BINARIZATION = "Binarization"
    REPAIR = "repair"

    def __init__(self, ruta):
        print(f"problema leido")

    def getNombre(self):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")
    
    def getNumDim(self):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")
        
    def getDominioDim(self):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")

    def evaluarFitness(self, decodedInstance):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")

    def setParametros(self, parametros):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")

    def getParametros(self, parametros):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")

    def encode(self, decodedInstances):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")

    def decode(self, encodedInstances):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")

    def generarSoluciones(self, numSoluciones):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")

    def repararSoluciones(self, soluciones):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")

    def getIndiceMejora(self):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")

    def getMejorEvaluacion(self):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")

    def getMejoresSoluciones(self):
        raise Exception(f"{inspect.currentframe().f_code.co_name} No implementado")