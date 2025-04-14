import os
import yaml
import simpy
import random
import time
import pandas as pd
import numpy as np
from collections import defaultdict

# Importamos las funciones de graficación desde plots.py
from plots import plot_occupancy, run_lambda_sweep, run_3d_graph

# ----------------------------------------------------------------------------
# Función para cargar la configuración desde un archivo YAML
# ----------------------------------------------------------------------------
def load_config(filename):
    """Carga la configuración desde un archivo YAML.
       Devuelve un diccionario con la configuración o uno por defecto en caso de error."""
    try:
        with open(filename, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        print(f"Error al cargar la configuración: {e}")
        # Configuración por defecto en caso de error
        return {
            "simulation": {
                "OW": 120,
                "IT": 1,
                "MIT": 5,
                "Tunit": 10,
                "duration": 1000
            },
            "event_generator": {
                "generators_per_unit": 30,
                "lambda": 0.5
            }
        }

# ----------------------------------------------------------------------------
# Clase SensorEvent
# ----------------------------------------------------------------------------
class SensorEvent:
    """
    Representa un evento generado por un sensor.
    Se utiliza el timestamp del evento para determinar el slot de inferencia
    (slot = tiempo del evento + Ventana de Observación [OW]).
    """
    def __init__(self, nombre, tiempo):
        self.nombre = nombre
        self.tiempo = tiempo  # Momento en el que se genera el evento

# ----------------------------------------------------------------------------
# Clase Sensor (Generador de eventos)
# ----------------------------------------------------------------------------
class Sensor:
    """
    Genera eventos siguiendo una distribución exponencial.
    Se garantiza que para cada unidad de tiempo básica (Tunit) se genere como máximo un evento.
    """
    def __init__(self, env, procesador, nombre, lambda_param, tunit):
        self.env = env
        self.procesador = procesador
        self.nombre = nombre
        self.lambda_param = lambda_param  # Parámetro λ para la distribución exponencial.
        self.tunit = tunit  # Unidad de tiempo (Tunit) en segundos.
        self.contador = 0
        self.last_event_time = -1  # Evita que se generen eventos en el mismo instante.
        env.process(self.generar_eventos())

    def generar_eventos(self):
        """Genera eventos de acuerdo a la distribución exponencial escalada por Tunit."""
        while True:
            tiempo_espera = self.tunit * int(random.expovariate(self.lambda_param))
            tiempo_espera = max(self.tunit, tiempo_espera)
            yield self.env.timeout(tiempo_espera)
            
            if self.env.now > self.last_event_time:
                self.contador += 1
                self.last_event_time = self.env.now
                evento = SensorEvent(f"{self.nombre}_ev{self.contador}", self.env.now)
                self.procesador.events_generated += 1
                self.procesador.regular_store.put((evento.nombre, evento.tiempo))

# ----------------------------------------------------------------------------
# Clase Procesador (Planificador de inferencias simplificado)
# ----------------------------------------------------------------------------
class Procesador:
    """
    Nodo procesador que gestiona la planificación de inferencias.
    Maneja dos canales: Regular y Offloading.
    """
    def __init__(self, env, nombre, config):
        self.env = env
        self.nombre = nombre
        self.config = config
        self.regular_store = simpy.Store(env)
        self.offloading_store = simpy.Store(env)
        
        # Diccionario para almacenar información de cada slot (clave: tiempo_inicial, valor: datos del slot)
        self.slots_info = {}
        
        # Monitores para graficar (listas de tuplas)
        self.regular_monitor = []    # (tiempo_actual, slot, uso_acumulado)
        self.offloading_monitor = [] # (tiempo_actual, slot, offload)
        
        # Contadores de eventos
        self.events_generated = 0
        self.events_processed = 0
        self.events_offloaded = 0
        
        env.process(self.process_regular())

    def process_regular(self):
        """Procesa los eventos en el canal regular, asignándolos a slots o offloadeándolos si es necesario."""
        while True:
            nombre_evento, tiempo_evento = yield self.regular_store.get()
            # Se calcula el tiempo inicial oficial para el slot
            tiempo_inicial_oficial = tiempo_evento + self.config["simulation"]["OW"]
            
            if tiempo_inicial_oficial not in self.slots_info:
                self.slots_info[tiempo_inicial_oficial] = {
                    "tiempo_inicial_oficial": tiempo_inicial_oficial,
                    "tiempo_final_oficial": tiempo_inicial_oficial + self.config["simulation"]["MIT"],
                    "n_inferencias": 0,
                    "tiempo_total_inferencias": 0
                }
            slot = self.slots_info[tiempo_inicial_oficial]
            
            # Verificar si hay capacidad en el slot
            if (tiempo_inicial_oficial + slot["tiempo_total_inferencias"] + self.config["simulation"]["IT"] <=
                slot["tiempo_final_oficial"]):
                slot["n_inferencias"] += 1
                slot["tiempo_total_inferencias"] += self.config["simulation"]["IT"]
                self.regular_monitor.append((self.env.now, tiempo_inicial_oficial, slot["tiempo_total_inferencias"]))
                self.events_processed += 1
            else:
                self.events_offloaded += 1
                self.offloading_monitor.append((self.env.now, tiempo_inicial_oficial, 1))
            
# ----------------------------------------------------------------------------
# Función para configurar la simulación
# ----------------------------------------------------------------------------
def setup_simulation(config):
    env = simpy.Environment()
    procesador = Procesador(env, "P1.1.1", config)
    
    # Crea los sensores según la cantidad especificada en la configuración
    for i in range(config["event_generator"]["generators_per_unit"]):
        Sensor(env, procesador, f"P1.1.1_sensor{i}",
               config["event_generator"]["lambda"],
               config["simulation"]["Tunit"])
    return env, procesador

# ----------------------------------------------------------------------------
# Función para ejecutar la simulación completa
# ----------------------------------------------------------------------------
def run_simulation(config):
    env, processor = setup_simulation(config)
    duration = config["simulation"]["duration"]
    
    start_time = time.time()
    env.run(until=duration)
    elapsed = time.time() - start_time
    
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
# Función principal
# ----------------------------------------------------------------------------
def main():
    # La configuración se carga desde config.yaml, que se encuentra en el mismo directorio
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, "config.yaml")
    config_global = load_config(config_path)
    
    print("Configuración actual:")
    print(f"  OW (Ventana de observación): {config_global['simulation']['OW']} s")
    print(f"  IT (Tiempo de inferencia): {config_global['simulation']['IT']} s")
    print(f"  MIT (Máximo tiempo de inferencia por slot): {config_global['simulation']['MIT']} s")
    print(f"  Tunit (Unidad de tiempo): {config_global['simulation']['Tunit']} s")
    print(f"  Duración: {config_global['simulation']['duration']} s")
    print(f"  Sensores por procesador: {config_global['event_generator']['generators_per_unit']}")
    print(f"  Lambda (λ): {config_global['event_generator']['lambda']}")
    print()
    
    _, processor, stats = run_simulation(config_global)
    print(f"Simulación completada en {stats['simulation_time']:.2f} segundos")
    print(f"Eventos Generados: {stats['events_generated']}")
    print(f"Eventos Aceptados: {stats['events_accepted']}")
    print(f"Eventos Offloaded: {stats['events_overflow']}")
    print(f"Porcentaje Offloading: {stats['overflow_percentage']:.2f}%")
    print()
    
    # Llamada a las funciones para graficar los resultados
    plot_occupancy(processor, config_global)
    run_lambda_sweep(config_global)
    run_3d_graph(config_global)

if __name__ == "__main__":
    main()
