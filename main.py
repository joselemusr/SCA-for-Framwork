import os
from datetime import datetime
from BD import RegistroMH
from Problema import LectorProbOpt
from Problema import ProblemaFactory
from MH import MHFactory
from Solver.GenericSolver import GenericSolver
from DTO import EstadoExperimento
from Agente import AgenteFactory
from MH.Metaheuristica import Metaheuristica as MH
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-E", "--experimento", help="Nombre del experimento, almacenado en BD", required=True)
ARGS = parser.parse_args()


C_PROBLEM = '.Problema/Instancias'

if __name__ == "__main__":
    
    nomExperimento = ARGS.experimento
    #print(f"args.experimento {ARGS.experimento}")
    #exit()


    inicio = datetime.now()
    cont = 0
    
    #SOLO PARA PRUEBAS
    RegistroMH.insertDummyExp(nomExperimento)
    #RegistroMH.insertDummyExpPSO(nomExperimento)

    for _ in range(100):
        experimento = RegistroMH.obtenerExperimento(nomExperimento)
        if experimento is None: break
        try:
            parametros = experimento.getParametros()
            if parametros.getInstProblema() is not None:
                problema = ProblemaFactory.crearConParams(parametros.getNomProblema(), parametros.getInstProblema())
                problema.setParametros({problema.TRANSFER_FUNCTION:parametros.getTransferFunctionType(),problema.BINARIZATION:parametros.getBinarizationType(),problema.REPAIR:parametros.getRepairType()})
            else:
                problema = ProblemaFactory.crear(parametros.getNomProblema())
            print(f"nombre del problema {problema.getNombre()}")
            
            

            mh = MHFactory.crear(parametros.getNomMH())
            
            mh.setProblema(problema)
            
            mh.setParametros(parametros.getParametrosMH())
            
            agente = AgenteFactory.crear(parametros.getNomAgente())
            agente.setParametrosAutonomos(parametros.getParametrosAgente())
            agente.setTotIter(mh.getParametros()[MH.NUM_ITER])
            solver = GenericSolver()
            solver.setMH(mh)
            solver.setAgente(agente)
            resultado = solver.resolverProblema(experimento.getId())
            experimento.setResultado(resultado)
            experimento.setEstado(EstadoExperimento.TERMINADO)
            fin = datetime.now()
            RegistroMH.guardarExperimento(experimento, inicio, fin)
            cont += 1
        except Exception as ex:
            experimento.setEstado(EstadoExperimento.PENDIENTE)
            fin = datetime.now()
            RegistroMH.guardarExperimento(experimento, inicio, fin)
            raise ex
    fin = datetime.now()
    print(f"fin, {cont} experimentos ejecutados, tiempo de ejecución: {fin-inicio}")