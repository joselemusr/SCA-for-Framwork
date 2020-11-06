class Parametro:
    def __init__(self):
        print("Construyendo un conjunto de parametros")
        self.nomProblema = None
        self.instProblema = None
        self.nomMH = None
        self.nomAgente = None
        self.parametrosAgente = None
        self.parametrosMH = None
        self.TransferFunctionType = None
        self.BinarizationType = None
        self.RepairType = None
    
    def setParametrosMH(self,parametros):
        self.parametrosMH = parametros

    def getParametrosMH(self):
        return self.parametrosMH

    def setParametrosAutonomo(self,ParametrosAutonomo):
        self.ParametrosAutonomo = ParametrosAutonomo

    def getParametrosAutonomo(self):
        return self.ParametrosAutonomo

    def setParametrosAgente(self,parametros):
        self.parametrosAgente = parametros

    def getParametrosAgente(self):
        return self.parametrosAgente

    def setNomProblema(self,nomProblema):
        self.nomProblema = nomProblema

    def getNomProblema(self):
        return self.nomProblema

    def setInstProblema(self,instProblema):
        self.instProblema = instProblema

    def getInstProblema(self):
        return self.instProblema

    def setNomMH(self,nomMH):
        self.nomMH = nomMH

    def getNomMH(self):
        return self.nomMH

    def setNomAgente(self,nomAgente):
        self.nomAgente = nomAgente

    def getNomAgente(self):
        return self.nomAgente

    def setTransferFunctionType(self,TransferFunctionType):
        self.TransferFunctionType = TransferFunctionType

    def getTransferFunctionType(self):
        return self.TransferFunctionType

    def setBinarizationType(self,BinarizationType):
        self.BinarizationType = BinarizationType

    def getBinarizationType(self):
        return self.BinarizationType

    def setRepairType(self,RepairType):
        self.RepairType = RepairType

    def getRepairType(self):
        return self.RepairType