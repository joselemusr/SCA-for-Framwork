from Problema.Problema import Problema

def leer(ruta):
    print(f"Obteniendo instancia desde {ruta}")
    return Problema(ruta)