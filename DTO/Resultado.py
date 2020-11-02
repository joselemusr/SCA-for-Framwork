class Resultado:
    def __init__(self):
        
        self.fitness = None
        self.mejorSolucion = None
        print(f"Instanciando resultado")

    def setFitness(self,fitness):
        self.fitness = fitness

    def getFitness(self):
        return self.fitness

    def setMejorSolucion(self, mejorSol):
        self.mejorSolucion = mejorSol

    def getMejorSolucion(self):
        return self.mejorSolucion