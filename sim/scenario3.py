import os
import yaml
import simpy
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

# ----------------------------------------------------------------------------
# Cargar configuración desde config.yaml
# ----------------------------------------------------------------------------
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
except Exception as e:
    print(f"Error al cargar config.yaml: {e}")
    exit()

# ----------------------------------------------------------------------------
# Clase Sensor (Generador de eventos)
# ----------------------------------------------------------------------------
class Sensor:
    def __init__(self, env, procesador, nombre, lambda_param=None, tunit=None):
        self.env = env
        self.procesador = procesador
        self.nombre = nombre
        self.contador = 0
        self.lambda_param = lambda_param or config["event_generator"]["lambda"]
        self.tunit = tunit or config["simulation"]["Tunit"]
        env.process(self.generar_eventos())

    def generar_eventos(self):
        """Genera eventos con espera exponencial escalada por Tunit."""
        while True:
            # Generar tiempo de espera exponencial continuo
            tiempo_base = random.expovariate(self.lambda_param)
            # Escalar y redondear hacia arriba al múltiplo de Tunit
            tiempo_espera = math.ceil(tiempo_base * self.tunit)
            # Asegurar mínimo de 1 Tunit
            tiempo_espera = max(self.tunit, tiempo_espera)

            yield self.env.timeout(tiempo_espera)

            self.contador += 1
            self.procesador.events_generated += 1
            nombre_evento = f"{self.nombre}_ev{self.contador}"
            tiempo_evento = self.env.now
            self.procesador.regular_store.put((nombre_evento, tiempo_evento))

# ----------------------------------------------------------------------------
# Clase Planificador/Procesador - Con parámetros MIT/IT variables
# ----------------------------------------------------------------------------
class Planificador:
    def __init__(self, env, nombre, mit=None, it=None, ow=None):
        self.env = env
        self.nombre = nombre
        self.regular_store = simpy.Store(env)
        self.slots_info = {}
        self.events_generated = 0
        self.events_processed = 0
        self.events_offloaded = 0
        self.regular_monitor = []
        self.offloading_monitor = []
        
        # Usar valores pasados o tomar del config
        self.MIT = mit if mit is not None else config["simulation"]["MIT"]
        self.IT = it if it is not None else config["simulation"]["IT"]
        self.OW = ow if ow is not None else config["simulation"]["OW"]
        
        env.process(self.procesar_eventos())

    def procesar_eventos(self):
        while True:
            nombre_evento, t0 = yield self.regular_store.get()
            t0_oficial = t0 + self.OW

            # 1) Calcular bloqueos entre ventanas
            inicio_real = t0_oficial
            for prev_inicio, datos in sorted(self.slots_info.items()):
                if datos['fin_real'] > inicio_real:
                    inicio_real = datos['fin_real']

            # 2) Registrar nueva ventana si no existe
            if t0_oficial not in self.slots_info:
                self.slots_info[t0_oficial] = {
                    'inicio_real': inicio_real,
                    'fin_oficial': t0_oficial + self.MIT,
                    'fin_real': inicio_real,
                    'sum_it': 0,
                    'n_inf': 0
                }

            ventana = self.slots_info[t0_oficial]

            # 3) Intentar aceptar inferencia según tiempo real disponible
            tiempo_disponible = ventana['fin_oficial'] - ventana['inicio_real']
            if ventana['sum_it'] + self.IT <= tiempo_disponible:
                ventana['n_inf'] += 1
                ventana['sum_it'] += self.IT
                ventana['fin_real'] = ventana['inicio_real'] + ventana['sum_it']
                self.events_processed += 1
                self.regular_monitor.append((self.env.now, t0_oficial, ventana['sum_it']))
            else:
                self.events_offloaded += 1
                self.offloading_monitor.append((self.env.now, t0_oficial, 1))

    def obtener_resultados(self):
        porcentaje = (self.events_offloaded / self.events_generated * 100) if self.events_generated > 0 else 0
        return {
            'events_generated': self.events_generated,
            'events_processed': self.events_processed,
            'events_offloaded': self.events_offloaded,
            'overflow_percentage': porcentaje,
            'mit_actual': self.MIT,
            'it_actual': self.IT,
            'mit_it_ratio': self.MIT / self.IT if self.IT > 0 else 0
        }

# Alias para compatibilidad
Procesador = Planificador

# ----------------------------------------------------------------------------
# Setup y simulación con parámetros variables
# ----------------------------------------------------------------------------
def setup_simulation(cfg, mit=None, it=None, ow=None):
    env = simpy.Environment()
    proc = Procesador(env, "P1", mit=mit, it=it, ow=ow)
    
    for i in range(cfg["event_generator"]["generators_per_unit"]):
        Sensor(env, proc,
               nombre=f"S{i}",
               lambda_param=cfg["event_generator"]["lambda"],
               tunit=cfg["simulation"]["Tunit"])
    return env, proc

def run_simulation(cfg, mit=None, it=None, ow=None):
    env, proc = setup_simulation(cfg, mit=mit, it=it, ow=ow)
    dur = cfg["simulation"]["duration"]
    inicio = time.time()
    env.run(until=dur)
    elapsed = time.time() - inicio

    stats = proc.obtener_resultados()
    stats['simulation_time'] = elapsed
    return env, proc, stats

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    # Importar aquí para evitar importación circular
    from plots3 import plot_occupancy, run_lambda_sweep, run_3d_graph
    
    print("Configuración:", config["simulation"], config["event_generator"])
    env, proc, stats = run_simulation(config)
    print("Stats:", stats)

    # Generar gráficas
    plot_occupancy(proc, config)
    run_lambda_sweep(config)
    run_3d_graph(config)

if __name__ == "__main__":
    main()