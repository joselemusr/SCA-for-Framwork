from DTO.Experimento import Experimento
from DTO.Parametro import Parametro
from DTO.ParametroAgente import ParametroAgente
from DTO.ParametroMH import ParametroMH
from DTO import TipoDominio, TipoComponente
from MH.PSO import PSO
from MH.SCA import SCA

from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sqlalchemy as db
import configparser
import json
from collections import namedtuple

config = configparser.ConfigParser()
config.read('BD/conf/db_config.ini')
host = config['postgres']['host']
db_name = config['postgres']['db_name']
port = config['postgres']['port']
user = config['postgres']['user']
pwd = config['postgres']['pass']

engine = create_engine(f'postgresql://{user}:{pwd}@{host}:{port}/{db_name}')

sqlObtenerExp  = "update datos_ejecucion set estado = 'ejecucion', inicio = :inicio "
sqlObtenerExp += "where id =  "
sqlObtenerExp += "(select id from datos_ejecucion "
sqlObtenerExp += "    where estado = 'pendiente' "
sqlObtenerExp += "    and nombre_algoritmo = :nombre "
sqlObtenerExp += "    order by id asc "
sqlObtenerExp += "    limit 1) returning id, parametros;"

def insertDummyExp(nombreExperimento):
    sqlInsert = "INSERT INTO datos_ejecucion(nombre_algoritmo, parametros, estado) VALUES (:nomExp, :param, 'pendiente')"
    parametros = Parametro()
    parametros.setNomProblema("SCP")
    #parametros.setInstProblema(paramBD.nomInstProblema)
    parametros.setNomMH("SCA")
    parametros.setInstProblema("Problema/scp/instances/mscpnrg2.txt")
    parametros.setTransferFunctionType("V4")
    parametros.setBinarizationType("Elitist")
    parametros.setRepairType("repairSimple") #Puede ser "repairGPU", "repairSimple" o "repairCompleja" hasta el momento
    parametros.setNomAgente("QLearning")
    paramsMH = {}
    saltoStr = "a"
    paramsMH[saltoStr] = 2
    paramsMH[SCA.NP] = 5
    paramsMH[SCA.NUM_ITER] = 10
    parametros.setParametrosMH(paramsMH)
    paramsAutonomos = []
    tBinaryName = "tBinary"
    tBinary = ParametroAgente()
    tBinary.setNombre(tBinaryName)
    tBinary.setTipo(TipoDominio.DISCRETO)
    tBinary.setMinimo(1)
    tBinary.setMaximo(40)
    tBinary.setPaso(1)
    tBinary.setComponente(TipoComponente.PROBLEMA)
    
    tTransferenciaName = "tTransferencia"
    tTransferencia = ParametroAgente()
    tTransferencia.setNombre(tTransferenciaName)
    tTransferencia.setTipo(TipoDominio.DISCRETO)
    tTransferencia.setMinimo(1)
    tTransferencia.setMaximo(40)
    tTransferencia.setPaso(1)
    tTransferencia.setComponente(TipoComponente.PROBLEMA)
    
    paramsAgente = {}
    Gamma = "Gamma"
    Actions = "Actions"
    stateType = "stateType"
    qlAlphaType = "qlAlphaType"
    rewardType = "rewardType"
    PolicyType = "PolicyType"
    iterMax = "iterMax"
    epsilon = "epsilon"
    qlAlpha = "qlAlpha"
    paramsAgente[Gamma] = 0.4
    paramsAgente[Actions] = 40
    paramsAgente[stateType] = 2
    paramsAgente[qlAlphaType] = "static"
    paramsAgente[rewardType] = "withPenalty1"
    paramsAgente[PolicyType] = "softMax-rulette-elitist"
    paramsAgente[iterMax] = 10
    paramsAgente[epsilon] = 0.1
    paramsAgente[qlAlpha] = 0.1
        
    parametros.setParametrosAgente(paramsAgente)


    session = createSession()
    session.execute(sqlInsert,{
        "nomExp":nombreExperimento
        , "param": json.dumps(parametros.__dict__, default=lambda o: o.__dict__) 
        })
    session.commit()

def insertDummyExpPSO(nombreExperimento):
    for _ in range(30):
        sqlInsert = "INSERT INTO datos_ejecucion(nombre_algoritmo, parametros, estado) VALUES (:nomExp, :param, 'pendiente')"
        parametros = Parametro()
        parametros.setNomProblema("SCP")
        parametros.setInstProblema("Problema/scp/instances/mscp41.txt")
        parametros.setNomMH("PSO")
        parametros.setNomAgente("AgenteQL_HMM")
        paramsMH = {}
        paramsMH[PSO.C1] = 2
        paramsMH[PSO.C2] = 2
        paramsMH[PSO.W] = 0.4
        paramsMH[PSO.MIN_V] = -10
        paramsMH[PSO.MAX_V] = 1
        paramsMH[PSO.NP] = 10
        paramsMH[PSO.NUM_ITER] = 400
        parametros.setParametrosMH(paramsMH)
        paramsAgente = []

        np = ParametroAgente()
        np.setNombre(PSO.NP)
        np.setTipo(TipoDominio.DISCRETO)
        np.setMinimo(5)
        np.setMaximo(100)
        np.setValorInicial(paramsMH[PSO.NP])
        np.setComponente(TipoComponente.METAHEURISTICA)
        c1 = ParametroAgente()
        c1.setNombre(PSO.C1)
        c1.setTipo(TipoDominio.CONTINUO)
        c1.setMinimo(0)
        c1.setMaximo(4)
        c1.setComponente(TipoComponente.METAHEURISTICA)
        c2 = ParametroAgente()
        c2.setNombre(PSO.C2)
        c2.setTipo(TipoDominio.CONTINUO)
        c2.setMinimo(0)
        c2.setMaximo(4)
        c2.setComponente(TipoComponente.METAHEURISTICA)
        maxV = ParametroAgente()
        maxV.setNombre(PSO.MAX_V)
        maxV.setTipo(TipoDominio.CONTINUO)
        maxV.setMinimo(0)
        maxV.setMaximo(2)
        maxV.setComponente(TipoComponente.METAHEURISTICA)
        inercia = ParametroAgente()
        inercia.setNombre(PSO.W)
        inercia.setTipo(TipoDominio.CONTINUO)
        inercia.setMinimo(0)
        inercia.setMaximo(1)
        inercia.setComponente(TipoComponente.METAHEURISTICA)
        paramsAgente.append(c1)
        paramsAgente.append(c2)
        paramsAgente.append(maxV)
        paramsAgente.append(inercia)
        paramsAgente.append(np)
        parametros.setParametrosAgente(paramsAgente)
        session = createSession()
        session.execute(sqlInsert,{
            "nomExp":nombreExperimento
            , "param": json.dumps(parametros.__dict__, default=lambda o: o.__dict__) 
            })
        session.commit()

def createSession():
    return sessionmaker(bind=engine)()

def obtenerExperimento(nombreExperimento):
    print(f"Obteniendo experimento desde la base de datos")
    session = createSession()
    inicio = datetime.now()
    arrResult = session.execute(sqlObtenerExp,{"inicio":inicio, "nombre": nombreExperimento}).fetchone()
    session.commit()
    if arrResult is None: return None
    paramBD = json.loads(arrResult.parametros, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    parametros = Parametro()
    parametros.setNomProblema(paramBD.nomProblema)
    parametros.setInstProblema(paramBD.instProblema)
    parametros.setNomMH(paramBD.nomMH)
    parametros.setNomAgente(paramBD.nomAgente)
    parametros.setTransferFunctionType(paramBD.TransferFunctionType)
    parametros.setBinarizationType(paramBD.BinarizationType)
    parametros.setRepairType(paramBD.RepairType)
    parametros.setParametrosMH(paramBD.parametrosMH._asdict())
    paramsAgente = []
    for pAgente in paramBD.parametrosAgente:
        param = ParametroAgente(pAgente)
        paramsAgente.append(param)
    parametros.setParametrosAgente(paramsAgente)
    experimento = Experimento()
    experimento.setParametros(parametros)
    experimento.setId(arrResult.id)
    return experimento

def guardarExperimento(experimento, inicio, fin):
    session = createSession()
    metadata = db.MetaData()
    resultadoEjecucion = db.Table('resultado_ejecucion', metadata, autoload=True, autoload_with=engine)
    datosEjecucion = db.Table('datos_ejecucion', metadata, autoload=True, autoload_with=engine)
    insertResultadoEjecucion =resultadoEjecucion.insert()
    updateDatosEjecucion = datosEjecucion.update().where(datosEjecucion.c.id == experimento.getId())
    if experimento.getResultado() is not None:
        session.execute(insertResultadoEjecucion, {
                    'id_ejecucion':experimento.getId()
                    ,'fitness' : int(experimento.getResultado().getFitness())
                    ,'inicio': inicio 
                    ,'fin': fin
                    ,'mejor_solucion' : experimento.getResultado().getMejorSolucion()
                    })
    session.execute(updateDatosEjecucion, {
        "fin": fin
        ,"estado": experimento.getEstado()
    })
    session.commit()
    print(f"Experimento guardado")

def guardaDatosIteracion(data):
    if data is None:
        print(f"Nada que guardar!")
        exit()
    session = createSession()
    metadata = db.MetaData()
    datosIteracion = db.Table('datos_iteracion', metadata, autoload=True, autoload_with=engine)
    insertDatosIteracion = datosIteracion.insert()
    session.execute(insertDatosIteracion, data)
    session.commit()