from DTO.Experimento import Experimento
from DTO.Parametro import Parametro
from DTO.ParametroAgente import ParametroAgente
from DTO.ParametroMH import ParametroMH
from DTO import TipoDominio, TipoComponente
from MH.PSO import PSO

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
    parametros.setNomProblema("Esfera")
    #parametros.setInstProblema(paramBD.nomInstProblema)
    parametros.setNomMH("Rand1Dimension")
    parametros.setNomAgente("AgenteGenerico")
    paramsMH = {}
    saltoStr = "salto"
    paramsMH[saltoStr] = 2
    paramsMH["poblacion"] = 40
    paramsMH["numIter"] = 100
    parametros.setParametrosMH(paramsMH)
    paramsAgente = []
    salto = ParametroAgente()
    salto.setNombre(saltoStr)
    salto.setTipo(TipoDominio.CONTINUO)
    salto.setMinimo(1)
    salto.setMaximo(10)
    salto.setComponente(TipoComponente.METAHEURISTICA)
    paramsAgente.append(salto)
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