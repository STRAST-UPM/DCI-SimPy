import os
import yaml
import simpy
import random
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Cargar configuración desde config.yaml
# ---------------------------
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# ---------------------------
# Clases Básicas
# ---------------------------
class Event:
    def __init__(self, name, time):
        self.name = name
        self.time = time  # Se usa para contar ventanas

class Inference:
    def __init__(self, name, tunit):
        self.name = name
        self.tunit = tunit  # Unidad de tiempo (Tunit) en la que se generó

class Processor:
    """
    Representa un nodo de la jerarquía.
    Se conserva el procesamiento de eventos (regular y offloading) y se añade el manejo de inferencias.
    Las inferencias se procesan por Tunit con una capacidad local definida como min(Tunit, MIT).
    Las que excedan esa capacidad se envían al nodo padre (offloading).
    """
    def __init__(self, env, name, max_windows, parent_regular_store, parent_offloading_store, parent_inference_handler=None):
        self.env = env
        self.name = name
        self.max_windows = max_windows
        self.regular_store = simpy.Store(env)
        self.offloading_store = simpy.Store(env)
        self.parent_regular_store = parent_regular_store
        self.parent_offloading_store = parent_offloading_store
        self.parent_inference_handler = parent_inference_handler

        # Contadores para eventos
        self.regular_window_time = defaultdict(lambda: 0)
        self.offloading_window_time = defaultdict(lambda: 0)
        self.regular_monitor = []    # (tiempo, valor) para canal regular
        self.offloading_monitor = [] # (tiempo, valor) para canal offloading

        # Parámetros y monitores para inferencias
        self.inference_capacity = min(config["simmulation"]["Tunit"], config["simmulation"]["MIT"])
        self.inference_generated_monitor = []   # (Tunit, cantidad generada)
        self.inference_processed_monitor = []   # (Tunit, cantidad procesada)
        self.inference_offloaded_monitor = []   # (Tunit, cantidad offloaded)

        # Inicia procesos para eventos
        env.process(self.process_regular())
        env.process(self.process_offloading())

    def process_regular(self):
        while True:
            ev = yield self.regular_store.get()
            # Enviar una copia al canal regular del nodo padre, si existe.
            if self.parent_regular_store is not None:
                self.parent_regular_store.put(Event(ev.name, ev.time))
            current = self.regular_window_time[ev.time] + self.offloading_window_time[ev.time]
            # Comprobaciones de capacidad en la ventana
            if current < self.max_windows:
                self.regular_window_time[ev.time] += 1
                self.regular_monitor.append((self.env.now, self.regular_window_time[ev.time]))
                # Simula el procesamiento durante window_duration sin bloquear el sistema
                self.env.process(self.window_process(ev, "regular"))
            else:
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
        # Simula el procesamiento durante window_duration
        yield self.env.timeout(config["processing_unit"]["window_duration"])
        # No se decrementa el contador para imitar el modelo original.

    def handle_inferences(self, n, tunit):
        """
        Procesa n inferencias correspondientes a la unidad de tiempo tunit.
        Cada inferencia tarda 1 segundo.
        Se procesan hasta la capacidad local (min(Tunit, MIT)); el resto se offloadea al nodo padre.
        """
        # Registra el número de inferencias generadas para este Tunit
        self.inference_generated_monitor.append((tunit, n))
        processed = min(n, self.inference_capacity)
        offloaded = n - processed if n > self.inference_capacity else 0
        yield self.env.timeout(processed * 1)
        self.inference_processed_monitor.append((tunit, processed))
        self.inference_offloaded_monitor.append((tunit, offloaded))
        if offloaded > 0 and self.parent_inference_handler is not None:
            self.env.process(self.parent_inference_handler(offloaded, tunit))

class EventGenerator:
    """
    Generador de eventos: espera un tiempo (según distribución exponencial) y envía un evento
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

class InferenceGenerator:
    """
    Generador de inferencias: cada Tunit (unidad de tiempo) genera un número de inferencias
    (entre 0 y 24, obtenido con distribución exponencial) que se disparan al "arrival time + OW".
    """
    def __init__(self, env, processor, name):
        self.env = env
        self.processor = processor
        self.name = name
        self.inference_count = 0
        env.process(self.generate())

    def generate(self):
        Tunit = config["simmulation"]["Tunit"]
        rate = config["inference_generator"]["lambda"]
        tunit_index = 0
        while True:
            # Espera el intervalo de generación Tunit (por ejemplo, 10 s)
            yield self.env.timeout(Tunit)
            # Genera un número de inferencias usando distribución exponencial, limitado a 24
            num_inferences = min(int(random.expovariate(rate)), 24)
            # Comprobación del tiempo restante antes de esperar OW:
            remaining = config["simmulation"]["duration"] - self.env.now
            if remaining > config["simmulation"]["OW"]:
                yield self.env.timeout(config["simmulation"]["OW"])
            else:
                yield self.env.timeout(max(0, remaining))
            # Lanza el procesamiento de las inferencias para este Tunit
            self.env.process(self.processor.handle_inferences(num_inferences, tunit_index))
            tunit_index += 1

# ---------------------------
# Helper para series temporales (para generar gráficas)
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
# Setup de la Simulación (jerarquía)
# ---------------------------
def setup_simulation():
    env = simpy.Environment()
    not_processed = simpy.Store(env)
    
    # Nodo raíz: P1 (Control Center)
    p1 = Processor(env, "P1", 200, None, not_processed, parent_inference_handler=None)
    # Nodos intermedios: P1.1 y P1.2 (Substations)
    p1_1 = Processor(env, "P1.1", 20, p1.regular_store, p1.offloading_store, parent_inference_handler=p1.handle_inferences)
    p1_2 = Processor(env, "P1.2", 20, p1.regular_store, p1.offloading_store, parent_inference_handler=p1.handle_inferences)
    # Nodos de borde (edge/fog)
    p1_1_1 = Processor(env, "P1.1.1", 10, p1_1.regular_store, p1_1.offloading_store, parent_inference_handler=p1_1.handle_inferences)
    p1_1_2 = Processor(env, "P1.1.2", 10, p1_1.regular_store, p1_1.offloading_store, parent_inference_handler=p1_1.handle_inferences)
    p1_2_1 = Processor(env, "P1.2.1", 10, p1_2.regular_store, p1_2.offloading_store, parent_inference_handler=p1_2.handle_inferences)
    p1_2_2 = Processor(env, "P1.2.2", 10, p1_2.regular_store, p1_2.offloading_store, parent_inference_handler=p1_2.handle_inferences)
    
    leaf_nodes = [p1_1_1, p1_1_2, p1_2_1, p1_2_2]
    # Inicializa generadores en nodos edge
    for node in leaf_nodes:
        # Se mantienen los generadores de eventos
        for i in range(config["event_generator"]["generators_per_unit"]):
            EventGenerator(env, node, f"{node.name}_{i}")
        # Se añade el generador de inferencias
        for i in range(config["inference_generator"]["generators_per_unit"]):
            InferenceGenerator(env, node, f"{node.name}_inf_{i}")
    
    return env, p1, p1_1, p1_2, leaf_nodes

# ---------------------------
# Graficar Resultados
# ---------------------------
def plot_results(p1, p1_1, p1_2, leaf_nodes):
    duration = config["simmulation"]["duration"]
    
    # Gráficas de Eventos (figura 1)
    df_events = pd.DataFrame({"t": list(range(duration + 1))})
    for node in leaf_nodes:
        df_events[node.name] = get_series(node.regular_monitor, duration)
    df_events["P1.1"] = get_series(p1_1.regular_monitor, duration)
    df_events["P1.2"] = get_series(p1_2.regular_monitor, duration)
    df_events["P1"] = get_series(p1.regular_monitor, duration)
    df_events["P1_off"] = get_series(p1.offloading_monitor, duration)
    df_events["P1.1_off"] = get_series(p1_1.offloading_monitor, duration)
    df_events["P1.2_off"] = get_series(p1_2.offloading_monitor, duration)
    
    fig1, (ax_off, ax_reg) = plt.subplots(2, 1, figsize=(21, 14))
    ax_off.plot(df_events["t"], df_events["P1_off"], label="P1", drawstyle="steps-post", color="blue")
    ax_off.plot(df_events["t"], df_events["P1.1_off"], label="P1.1", drawstyle="steps-post", color="red")
    ax_off.plot(df_events["t"], df_events["P1.2_off"], label="P1.2", drawstyle="steps-post", color="green")
    ax_off.set_title("Offloading units (Eventos)")
    ax_off.set_xlabel("Tiempo (s)")
    ax_off.set_ylabel("Ventanas ocupadas")
    ax_off.legend(fontsize=14, loc="upper right")
    
    ax_reg.plot(df_events["t"], df_events["P1.1.1"], label="P1.1.1", drawstyle="steps-post")
    ax_reg.plot(df_events["t"], df_events["P1.1.2"], label="P1.1.2", drawstyle="steps-post")
    ax_reg.plot(df_events["t"], df_events["P1.2.1"], label="P1.2.1", drawstyle="steps-post")
    ax_reg.plot(df_events["t"], df_events["P1.2.2"], label="P1.2.2", drawstyle="steps-post")
    ax_reg.plot(df_events["t"], df_events["P1.1"], label="P1.1", drawstyle="steps-post", color="red")
    ax_reg.plot(df_events["t"], df_events["P1.2"], label="P1.2", drawstyle="steps-post", color="green")
    ax_reg.plot(df_events["t"], df_events["P1"], label="P1", drawstyle="steps-post", color="blue")
    ax_reg.set_title("Unidades regulares (Eventos)")
    ax_reg.set_xlabel("Tiempo (s)")
    ax_reg.set_ylabel("Ventanas ocupadas")
    ax_reg.legend(fontsize=14, loc="upper right")
    
    fig1.tight_layout()
    
    # Gráfica de Inferencias (figura 2)
    inf_generated = defaultdict(int)
    inf_processed = defaultdict(int)
    inf_offloaded = defaultdict(int)
    for node in leaf_nodes:
        for t, count in node.inference_generated_monitor:
            inf_generated[t] += count
        for t, count in node.inference_processed_monitor:
            inf_processed[t] += count
        for t, count in node.inference_offloaded_monitor:
            inf_offloaded[t] += count
    Tunit = config["simmulation"]["Tunit"]
    num_tunits = int(duration // Tunit) + 1
    t_units = list(range(num_tunits))
    gen_series = [inf_generated[t] if t in inf_generated else 0 for t in t_units]
    proc_series = [inf_processed[t] if t in inf_processed else 0 for t in t_units]
    off_series = [inf_offloaded[t] if t in inf_offloaded else 0 for t in t_units]
    
    fig2, ax_inf = plt.subplots(1, 1, figsize=(14, 8))
    ax_inf.plot(t_units, gen_series, label="Inferencias Generadas", marker="o", linestyle="-")
    ax_inf.plot(t_units, proc_series, label="Inferencias Procesadas (Edge)", marker="o", linestyle="--")
    ax_inf.plot(t_units, off_series, label="Inferencias Offloaded", marker="o", linestyle=":")
    ax_inf.set_title("Inferencias por Tunit")
    ax_inf.set_xlabel("Índice de Tunit")
    ax_inf.set_ylabel("Número de Inferencias")
    ax_inf.legend(fontsize=14, loc="upper right")
    fig2.tight_layout()
    
    # Guarda las gráficas en la carpeta "graphs"
    graphs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'graphs')
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)
    
    fig1.savefig(os.path.join(graphs_dir, f"events_sim_lambda_{config['event_generator']['lambda']}.svg"))
    fig2.savefig(os.path.join(graphs_dir, f"inferences_sim_lambda_{config['event_generator']['lambda']}.svg"))
    
    plt.show()

# ---------------------------
# Ejecutar la Simulación
# ---------------------------
def run_simulation():
    env, p1, p1_1, p1_2, leaf_nodes = setup_simulation()
    env.run(until=config["simmulation"]["duration"])
    plot_results(p1, p1_1, p1_2, leaf_nodes)

if __name__ == "__main__":
    run_simulation()
