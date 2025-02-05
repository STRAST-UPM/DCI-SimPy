import os
import yaml
import simpy
import random
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Load configuration from config.yaml
# ---------------------------
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# ---------------------------
# Basic Classes
# ---------------------------
class Event:
    def __init__(self, name, time):
        self.name = name
        self.time = time  # Se usa como clave para contar ventanas

class Processor:
    """
    Representa un nodo de la jerarquía.
    Cada nodo tiene dos stores (regular y offloading), un máximo de ventanas (max_windows)
    y dos diccionarios que cuentan las ventanas ocupadas (por instante de llegada).
    Si no hay capacidad para procesar el evento, se envía al canal offloading del nodo padre.
    """
    def __init__(self, env, name, max_windows, parent_regular_store, parent_offloading_store):
        self.env = env
        self.name = name
        self.max_windows = max_windows
        self.regular_store = simpy.Store(env)
        self.offloading_store = simpy.Store(env)
        self.parent_regular_store = parent_regular_store
        self.parent_offloading_store = parent_offloading_store

        self.regular_window_time = defaultdict(lambda: 0)
        self.offloading_window_time = defaultdict(lambda: 0)
        self.regular_monitor = []    # Lista de (tiempo, valor) para el canal regular
        self.offloading_monitor = [] # Lista de (tiempo, valor) para el canal offloading

        # Inicia los procesos de cada canal.
        env.process(self.process_regular())
        env.process(self.process_offloading())

    def process_regular(self):
        while True:
            ev = yield self.regular_store.get()
            # Enviar una copia al canal regular del nodo padre, si existe.
            if self.parent_regular_store is not None:
                self.parent_regular_store.put(Event(ev.name, ev.time))
            current = self.regular_window_time[ev.time] + self.offloading_window_time[ev.time]
            if current < self.max_windows:
                self.regular_window_time[ev.time] += 1
                self.regular_monitor.append((self.env.now, self.regular_window_time[ev.time]))
                # Simula el procesamiento esperando window_duration (la ventana permanece ocupada)
                self.env.process(self.window_process(ev, "regular"))
            else:
                # Si no hay capacidad, se envía al canal offloading del nodo padre, si existe
                if self.parent_offloading_store is not None:
                    self.parent_offloading_store.put(ev)

    def process_offloading(self):
        while True:
            ev = yield self.offloading_store.get()
            current = self.regular_window_time[ev.time] + self.offloading_window_time[ev.time]
            if current < self.max_windows:
                self.offloading_window_time[ev.time] += 1
                self.offloading_monitor.append((self.env.now, self.offloading_window_time[ev.time]))
                self.env.process(self.window_process(ev, "offloading"))
            else:
                if self.parent_offloading_store is not None:
                    self.parent_offloading_store.put(ev)

    def window_process(self, ev, channel):
        # Simula el procesamiento durante el tiempo de la ventana.
        yield self.env.timeout(config["processing_unit"]["window_duration"])
        # No se decrementa el contador para imitar el modelo original.

class EventGenerator:
    """
    Generador de eventos: espera un tiempo (exponencial) y envía un evento
    al store regular del procesador asociado.
    """
    def __init__(self, env, processor, name):
        self.env = env
        self.processor = processor
        self.name = name
        self.event_count = 0
        env.process(self.generate())

    def generate(self):
        while True:
            interarrival = int(random.expovariate(config["event_generator"]["lambda"]))
            yield self.env.timeout(interarrival)
            ev = Event(f"{self.name}_ev{self.event_count}", self.env.now)
            self.event_count += 1
            self.processor.regular_store.put(ev)

# ---------------------------
# Setup the Simulation Hierarchy
# ---------------------------
def setup_simulation():
    env = simpy.Environment()
    not_processed = simpy.Store(env)
    
    # Nodo raíz: P1 (Control Center)
    p1 = Processor(env, "P1", 200, None, not_processed)
    # Nodos intermedios: P1.1 y P1.2 (Substations)
    p1_1 = Processor(env, "P1.1", 20, p1.regular_store, p1.offloading_store)
    p1_2 = Processor(env, "P1.2", 20, p1.regular_store, p1.offloading_store)
    # Nodos de borde (edge/fog)
    p1_1_1 = Processor(env, "P1.1.1", 10, p1_1.regular_store, p1_1.offloading_store)
    p1_1_2 = Processor(env, "P1.1.2", 10, p1_1.regular_store, p1_1.offloading_store)
    p1_2_1 = Processor(env, "P1.2.1", 10, p1_2.regular_store, p1_2.offloading_store)
    p1_2_2 = Processor(env, "P1.2.2", 10, p1_2.regular_store, p1_2.offloading_store)
    
    # Inicializa generadores en cada nodo edge.
    leaf_nodes = [p1_1_1, p1_1_2, p1_2_1, p1_2_2]
    for node in leaf_nodes:
        for i in range(config["event_generator"]["generators_per_unit"]):
            EventGenerator(env, node, f"{node.name}_{i}")
    
    return env, p1, p1_1, p1_2, leaf_nodes

# ---------------------------
# Helper to Build Time Series from Monitor Data
# ---------------------------
def get_series(monitor, duration):
    series = []
    monitor_sorted = sorted(monitor, key=lambda x: x[0])
    current = 0
    idx = 0
    for t in range(duration + 1):
        while idx < len(monitor_sorted) and monitor_sorted[idx][0] <= t:
            current = monitor_sorted[idx][1]
            idx += 1
        series.append(current)
    return series

# ---------------------------
# Plotting the Results
# ---------------------------
def plot_results(p1, p1_1, p1_2, leaf_nodes):
    duration = config["simmulation"]["duration"]
    df = pd.DataFrame({"t": list(range(duration + 1))})
    
    # Agregar series para los nodos edge (canal regular)
    for node in leaf_nodes:
        df[node.name] = get_series(node.regular_monitor, duration)
    df["P1.1"] = get_series(p1_1.regular_monitor, duration)
    df["P1.2"] = get_series(p1_2.regular_monitor, duration)
    df["P1"] = get_series(p1.regular_monitor, duration)
    
    # Agregar series para los canales offloading en los nodos superiores
    df["P1_off"] = get_series(p1.offloading_monitor, duration)
    df["P1.1_off"] = get_series(p1_1.offloading_monitor, duration)
    df["P1.2_off"] = get_series(p1_2.offloading_monitor, duration)
    
    fig, (ax_off, ax_reg) = plt.subplots(2, 1, figsize=(21, 14))
    
    # Gráfico para Offloading units (arriba)
    ax_off.plot(df["t"], df["P1_off"], label="P1", drawstyle="steps-post", color="blue")
    ax_off.plot(df["t"], df["P1.1_off"], label="P1.1", drawstyle="steps-post", color="red")
    ax_off.plot(df["t"], df["P1.2_off"], label="P1.2", drawstyle="steps-post", color="green")
    ax_off.set_title("Offloading units")
    ax_off.set_xlabel("Time (s)")
    ax_off.set_ylabel("Occupied Windows")
    ax_off.legend(fontsize=14, loc="upper right")
    
    # Gráfico para Regular units (abajo)
    ax_reg.plot(df["t"], df["P1.1.1"], label="P1.1.1", drawstyle="steps-post")
    ax_reg.plot(df["t"], df["P1.1.2"], label="P1.1.2", drawstyle="steps-post")
    ax_reg.plot(df["t"], df["P1.2.1"], label="P1.2.1", drawstyle="steps-post")
    ax_reg.plot(df["t"], df["P1.2.2"], label="P1.2.2", drawstyle="steps-post")
    ax_reg.plot(df["t"], df["P1.1"], label="P1.1", drawstyle="steps-post", color="red")
    ax_reg.plot(df["t"], df["P1.2"], label="P1.2", drawstyle="steps-post", color="green")
    ax_reg.plot(df["t"], df["P1"], label="P1", drawstyle="steps-post", color="blue")
    ax_reg.set_title("Regular units")
    ax_reg.set_xlabel("Time (s)")
    ax_reg.set_ylabel("Occupied Windows")
    ax_reg.legend(fontsize=14, loc="upper right")
    
    fig.subplots_adjust(hspace=0.3)
    
    # Crear la carpeta "graphs" si no existe
    graphs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'graphs')
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)
    
    graph_filename = os.path.join(graphs_dir, f"sim_lambda_{config['event_generator']['lambda']}.svg")
    fig.savefig(graph_filename)
    plt.show()

# ---------------------------
# Run the Simulation
# ---------------------------
def run_simulation():
    env, p1, p1_1, p1_2, leaf_nodes = setup_simulation()
    env.run(until=config["simmulation"]["duration"])
    plot_results(p1, p1_1, p1_2, leaf_nodes)

if __name__ == "__main__":
    run_simulation()
