import os
import yaml
import simpy
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

# ---------------------------
# Cargar configuración desde config.yaml
# ---------------------------
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
except FileNotFoundError:
    print(f"Error: No se encontró config.yaml en {config_path}")
    exit()
except Exception as e:
    print(f"Error al cargar config.yaml: {e}")
    exit()

# Importar las funciones de generación de gráficas desde plots.py
from plots import plot_occupancy, run_lambda_sweep, run_3d_graph

# ----------------------------------------------------------------------------
# Clase Sensor (Generador de eventos)
# ----------------------------------------------------------------------------
class Sensor:
    """
    Representa un sensor que genera eventos siguiendo una distribución exponencial.
    """
    def __init__(self, env, procesador, nombre, lambda_param=None, tunit=None):
        self.env = env
        self.procesador = procesador
        self.nombre = nombre
        self.contador = 0
        
        # Use passed parameters or config values
        self.lambda_param = lambda_param if lambda_param is not None else config["event_generator"]["lambda"]
        self.tunit = tunit if tunit is not None else config["simulation"]["Tunit"]
        
        env.process(self.generar_eventos())

    def generar_eventos(self):
        """Genera eventos con la temporización original y los envía al procesador."""
        while True:
            # Lógica de Tiempo de Espera
            tiempo_base = random.expovariate(self.lambda_param)
            tiempo_espera_discreto = self.tunit * int(tiempo_base)
            tiempo_espera = max(self.tunit, tiempo_espera_discreto)
           
            # Pausa hasta el próximo evento
            yield self.env.timeout(tiempo_espera)
           
            # Generación del evento
            self.contador += 1
            self.procesador.events_generated += 1
            nombre_evento = f"{self.nombre}_ev{self.contador}"
            tiempo_actual_t0 = self.env.now
            self.procesador.regular_store.put((nombre_evento, tiempo_actual_t0))

# ----------------------------------------------------------------------------
# Clase Planificador/Procesador (Implementa la lógica del planificador descrito)
# ----------------------------------------------------------------------------
class Planificador:
    """
    Planificador que gestiona eventos y decide si aceptarlos o generar offloading
    según el algoritmo descrito en la documentación.
    """
    def __init__(self, env, nombre):
        self.env = env
        self.nombre = nombre
        self.regular_store = simpy.Store(env)
       
        # Estructura de datos para la tabla de eventos
        self.tabla_eventos = {}  # Diccionario para gestionar eventos por tiempo inicial
       
        # Contadores para estadísticas
        self.events_generated = 0
        self.events_processed = 0
        self.events_offloaded = 0
       
        # Registros para monitoreo
        self.monitor = []
        
        # Monitores adicionales para gráficas
        self.regular_monitor = []
        self.offloading_monitor = []
       
        # Iniciar proceso de procesamiento
        env.process(self.procesar_eventos())
   
    def procesar_eventos(self):
        """Implementa el algoritmo del planificador para procesar eventos."""
        OW = config["simulation"]["OW"]       # Ventana de observación
        IT = config["simulation"]["IT"]       # Tiempo de inferencia por evento
        MIT = config["simulation"]["MIT"]     # Máximo Intervalo de Tiempo permitido
        
        # Variables para monitores
        regular_slots = 0
        offloading_slots = 0
       
        while True:
            # Esperar y obtener el próximo evento
            nombre_evento, tiempo_evento_t0 = yield self.regular_store.get()
           
            # Calcular tiempo inicial oficial (t0 + OW)
            tiempo_inicial_oficial = tiempo_evento_t0 + OW
           
            # Verificar si hay bloqueo por eventos anteriores
            tiempo_bloqueado = 0
            tiempo_inicial_real = tiempo_inicial_oficial
           
            # Buscar posibles bloqueos de eventos anteriores
            for t_prev, datos in sorted(self.tabla_eventos.items()):
                if t_prev < tiempo_inicial_oficial:  # Solo eventos anteriores
                    # Verificar condición de bloqueo:
                    # t0_anterior + W + sumatorio_IT_anterior > t0_actual + W
                    tiempo_final_anterior = t_prev + datos['sumatorio_it']
                    if tiempo_final_anterior > tiempo_inicial_oficial:
                        # Hay bloqueo, ajustar tiempo inicial real
                        tiempo_inicial_real = tiempo_final_anterior
                        tiempo_bloqueado = tiempo_final_anterior - tiempo_inicial_oficial
           
            # Verificar si existe entrada en la tabla para este tiempo inicial oficial
            if tiempo_inicial_oficial not in self.tabla_eventos:
                # Crear nueva entrada en la tabla
                self.tabla_eventos[tiempo_inicial_oficial] = {
                    'tiempo_inicial_real': tiempo_inicial_real,
                    'tiempo_final_oficial': tiempo_inicial_oficial + MIT,
                    'tiempo_final_real': tiempo_inicial_real + MIT,
                    'n_inferencias': 0,
                    'sumatorio_it': 0
                }
           
            # Obtener entrada actual de la tabla
            entrada = self.tabla_eventos[tiempo_inicial_oficial]
           
            # Verificar condición de aceptación:
            # tiempo_inicial_real + sumatorio_IT + nuevo_IT < tiempo_inicial_oficial + MIT
            if entrada['tiempo_inicial_real'] + entrada['sumatorio_it'] + IT <= tiempo_inicial_oficial + MIT:
                # Aceptar evento
                entrada['n_inferencias'] += 1
                entrada['sumatorio_it'] += IT
                self.events_processed += 1
                
                # Actualizar monitor regular
                regular_slots += 1
                self.regular_monitor.append((self.env.now, tiempo_inicial_oficial, regular_slots))
               
                # Registrar en el monitor
                self.monitor.append({
                    'tiempo_decision': self.env.now,
                    'nombre_evento': nombre_evento,
                    'tiempo_evento': tiempo_evento_t0,
                    'tiempo_inicial_oficial': tiempo_inicial_oficial,
                    'tiempo_inicial_real': entrada['tiempo_inicial_real'],
                    'tiempo_bloqueado': tiempo_bloqueado,
                    'inferencias_aceptadas': entrada['n_inferencias'],
                    'tiempo_total_inferencias': entrada['sumatorio_it'],
                    'resultado': 'aceptado'
                })
            else:
                # Rechazar evento (offloading)
                self.events_offloaded += 1
                
                # Actualizar monitor offloading
                offloading_slots += 1
                self.offloading_monitor.append((self.env.now, tiempo_inicial_oficial, offloading_slots))
               
                # Registrar en el monitor
                self.monitor.append({
                    'tiempo_decision': self.env.now,
                    'nombre_evento': nombre_evento,
                    'tiempo_evento': tiempo_evento_t0,
                    'tiempo_inicial_oficial': tiempo_inicial_oficial,
                    'tiempo_inicial_real': entrada['tiempo_inicial_real'],
                    'tiempo_bloqueado': tiempo_bloqueado,
                    'inferencias_aceptadas': entrada['n_inferencias'],
                    'tiempo_total_inferencias': entrada['sumatorio_it'],
                    'resultado': 'offloading'
                })

    def obtener_resultados(self):
        """Calcula y devuelve las estadísticas finales."""
        offload_perc = (self.events_offloaded / self.events_generated * 100) if self.events_generated > 0 else 0
        return {
            "generados": self.events_generated,
            "procesados": self.events_processed,
            "offloaded": self.events_offloaded,
            "offload_percentage": offload_perc
        }

# Alias para mantener compatibilidad con ambas nomenclaturas
Procesador = Planificador

# ----------------------------------------------------------------------------
# Función auxiliar para construir series temporales a partir de monitor
# ----------------------------------------------------------------------------
def get_series(monitor, duration, key_idx=2):
    """
    A partir de la lista monitor (con tuplas de (tiempo, slot, uso_acumulado)),
    se construye una serie de tiempo (lista) en la que para cada instante t se registra
    el valor acumulado (uso) correspondiente.
    """
    series = []
    monitor_ordenado = sorted(monitor, key=lambda x: x[0])
    current_value = 0
    idx = 0
    for t in range(duration + 1):
        while idx < len(monitor_ordenado) and monitor_ordenado[idx][0] <= t:
            # Se actualiza el valor con la última medición del uso acumulado
            current_value = monitor_ordenado[idx][key_idx]
            idx += 1
        series.append(current_value)
    return series

# ----------------------------------------------------------------------------
# Función para configurar la simulación
# ----------------------------------------------------------------------------
def setup_simulation():
    env = simpy.Environment()
    procesador = Procesador(env, "P1.1.1")
    # Crear sensores de acuerdo al número indicado en config (generators_per_unit)
    for i in range(config["event_generator"]["generators_per_unit"]):
        Sensor(env, procesador, f"P1.1.1_sensor{i}", config["event_generator"]["lambda"], config["simulation"]["Tunit"])
    return env, procesador

# ----------------------------------------------------------------------------
# Función para ejecutar la simulación completa
# ----------------------------------------------------------------------------
def run_simulation(config):
    env, processor = setup_simulation()
    duration = config["simulation"]["duration"]
   
    # Ejecutar la simulación
    start_time = time.time()
    env.run(until=duration)
    elapsed = time.time() - start_time
   
    # Calcular estadísticas
    overflow_percentage = 0
    if processor.events_generated > 0:
        overflow_percentage = (processor.events_offloaded / processor.events_generated) * 100
   
    stats = {
        'events_generated': processor.events_generated,
        'events_accepted': processor.events_processed,
        'events_overflow': processor.events_offloaded,
        'overflow_percentage': overflow_percentage,
        'simulation_time': elapsed
    }
   
    return env, processor, stats

# ----------------------------------------------------------------------------
# Función principal para ejecutar la simulación y graficar
# ----------------------------------------------------------------------------
def main():
    # Intentamos cargar la configuración, o usamos la incorporada si falla
    try:
        config_global = config
    except NameError:
        config_global = None
    
    # Mostrar la configuración actual
    print("Configuración actual:")
    print(f"  OW (Ventana de observación): {config_global['simulation']['OW']} s")
    print(f"  IT (Tiempo de inferencia por evento): {config_global['simulation']['IT']} s")
    print(f"  MIT (Máximo tiempo de inferencia por slot): {config_global['simulation']['MIT']} s")
    print(f"  Tunit (Unidad de tiempo para generación de eventos): {config_global['simulation']['Tunit']} s")
    print(f"  Duración de la simulación: {config_global['simulation']['duration']} s")
    print(f"  Sensores por procesador: {config_global['event_generator']['generators_per_unit']}")
    print(f"  Lambda (λ): {config_global['event_generator']['lambda']}")
    print()
   
    # Ejecuta la simulación base y muestra estadísticas
    _, processor, stats = run_simulation(config_global)
    print(f"Simulación completada en {stats['simulation_time']:.2f} segundos")
    print(f"Eventos Generados: {stats['events_generated']}")
    print(f"Eventos Procesados (Aceptados): {stats['events_accepted']}")
    print(f"Eventos Offloaded: {stats['events_overflow']}")
    print(f"Porcentaje de Offloading: {stats['overflow_percentage']:.2f}%")
    print()
   
    # Generar las tres gráficas mejoradas
    plot_occupancy(processor, config_global)
    run_lambda_sweep(config_global)
    run_3d_graph(config_global)

# Ejecutar simulación si se ejecuta como script principal
if __name__ == "__main__":
    print("Iniciando simulación del Planificador...")
    main()
