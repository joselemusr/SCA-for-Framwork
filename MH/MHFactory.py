import importlib

def crear(nombreMh):
    print(f"Creando instancia de MH: {nombreMh}")
    
    module = importlib.import_module(f"MH.{nombreMh}")
    class_ = getattr(module, nombreMh)
    return class_()