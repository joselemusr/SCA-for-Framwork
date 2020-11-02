import importlib

def crear(nombreAgente):
    print(f"Creando instancia de Agente: {nombreAgente}")
    
    module = importlib.import_module(f"Agente.{nombreAgente}")
    class_ = getattr(module, nombreAgente)
    return class_()