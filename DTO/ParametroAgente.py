class ParametroAgente:

    def __init__(self, obj=None):
        if obj is None:
            self.tipo = None
            self.paso = None
            self.minimo = None
            self.maximo = None
            self.componente = None
            self.valorInicial = None
            self.nombre = None
        
        else:
            self.tipo = obj.tipo
            self.paso = obj.paso
            self.minimo = obj.minimo
            self.maximo = obj.maximo
            self.componente = obj.componente
            self.valorInicial = obj.valorInicial
            self.nombre = obj.nombre
        print(f"ParametroAgente creado")

    def setNombre(self, nombre):
        self.nombre = nombre

    def getNombre(self):
        return self.nombre

    def setValorInicial(self, valorInicial):
        self.valorInicial = valorInicial

    def getValorInicial(self):
        return self.valorInicial

    def setTipo(self, tipo):
        self.tipo = tipo

    def getTipo(self):
        return self.tipo

    def setPaso(self, paso):
        self.paso = paso

    def getPaso(self):
        return self.paso

    def setMinimo(self, minimo):
        self.minimo = minimo

    def getMinimo(self):
        return self.minimo

    def setMaximo(self, maximo):
        self.maximo = maximo

    def getMaximo(self):
        return self.maximo

    def setComponente(self, componente):
        self.componente = componente

    def getComponente(self):
        return self.componente