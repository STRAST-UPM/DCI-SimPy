import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------------------------------------------------------
# Función para obtener (y crear si no existe) el folder de guardado
# ----------------------------------------------------------------------------
def get_save_folder(config):
    mit    = config["simulation"]["MIT"]
    it     = config["simulation"]["IT"]
    tunit  = config["simulation"]["Tunit"]
    folder_name = f"MIT_{mit}_IT_{it}_Tunit_{tunit}"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(project_root, "graphs", folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

# ----------------------------------------------------------------------------
# Función auxiliar para construir series temporales a partir del monitor
# ----------------------------------------------------------------------------
def get_series(monitor, duration, key_idx=2):
    series = []
    monitor_ordenado = sorted(monitor, key=lambda x: x[0])
    current_value = 0
    idx = 0
    for t in range(duration + 1):
        while idx < len(monitor_ordenado) and monitor_ordenado[idx][0] <= t:
            current_value = monitor_ordenado[idx][key_idx]
            idx += 1
        series.append(current_value)
    return series

# ----------------------------------------------------------------------------
# Gráfica 1: Ventanas de Ocupación (Regular y Offloading)
# ----------------------------------------------------------------------------
def plot_occupancy(processor, config):
    duration = config["simulation"]["duration"]
    lambda_value = config["event_generator"]["lambda"]
    
    regular_series   = get_series(processor.regular_monitor, duration)
    offloading_series = get_series(processor.offloading_monitor, duration)
    t = range(duration + 1)
    
    y_max = max(max(regular_series, default=0), max(offloading_series, default=0))
    y_max = max(20, int(y_max * 1.2))
    
    fig, (ax_off, ax_reg) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Window Occupancy (λ = {lambda_value})", fontsize=16)
    
    ax_off.plot(t, offloading_series, label="Offloading",
                drawstyle="steps-post", color="red", linewidth=1.5)
    ax_off.set_title("Offloading Units")
    ax_off.set_ylabel("Windows", fontweight='bold')
    ax_off.set_ylim(0, y_max)
    ax_off.grid(True, linestyle='--', alpha=0.7)
    ax_off.legend(loc='best')
    
    ax_reg.plot(t, regular_series, label="Regular",
                drawstyle="steps-post", color="blue", linewidth=1.5)
    ax_reg.set_title("Regular Units")
    ax_reg.set_xlabel("Time (s)", fontweight='bold')
    ax_reg.set_ylabel("Windows", fontweight='bold')
    ax_reg.set_ylim(0, y_max)
    ax_reg.grid(True, linestyle='--', alpha=0.7)
    ax_reg.legend(loc='best')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_folder = get_save_folder(config)
    filename = os.path.join(save_folder, f"window_occupancy_lambda_{lambda_value}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Graph saved as: {filename}")

# ----------------------------------------------------------------------------
# Gráfica 2: Barrido de Lambda – % Offload vs Lambda (Lineal)
# ----------------------------------------------------------------------------
def run_lambda_sweep(config):
    from scenario import run_simulation

    duration = config["simulation"]["duration"]
    lambda_values = [0.001] + list(np.arange(0.05, 1.05, 0.05))
    lambda_values = sorted(set(lambda_values))
    
    overflow_percentages = []
    results_data = []
    original_lambda = config["event_generator"]["lambda"]

    print("\nRunning lambda sweep simulation...")
    for lam in lambda_values:
        config["event_generator"]["lambda"] = lam
        _, processor, stats = run_simulation(config)
        
        overflow_percentage = stats['overflow_percentage']
        overflow_percentages.append(overflow_percentage)
        
        results_data.append({
            'Lambda': lam,
            'Overflow (%)': overflow_percentage,
            'Events Generated': stats['events_generated'],
            'Events Accepted': stats['events_accepted'],
            'Events Overflow': stats['events_overflow']
        })
        
        print(f"Lambda: {lam:.3f}, Overflow %: {overflow_percentage:.2f}%")
    
    save_folder = get_save_folder(config)
    csv_file = os.path.join(save_folder, "lambda_sweep_results.csv")
    pd.DataFrame(results_data).to_csv(csv_file, index=False)
    print(f"Results saved to: {csv_file}")
    
    config["event_generator"]["lambda"] = original_lambda
    
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.plot(lambda_values, overflow_percentages, marker='o', linestyle='-',
            color='green', linewidth=2, markersize=8, label='Overflow %')
    
    z = np.polyfit(lambda_values, overflow_percentages, 3)
    p = np.poly1d(z)
    x_smooth = np.linspace(min(lambda_values), max(lambda_values), 100)
    y_smooth = p(x_smooth)
    ax.plot(x_smooth, y_smooth, 'r--', linewidth=1, alpha=0.7,
            label='Trend (3rd degree poly)')
    
    equation = f"y = {z[0]:.6f}x³ + {z[1]:.4f}x² + {z[2]:.4f}x + {z[3]:.4f}"
    ax.text(0.95, 0.05, equation, transform=ax.transAxes, fontsize=10,
            va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    for i in [0, len(lambda_values)//2, -1]:
        ax.annotate(f'{overflow_percentages[i]:.1f}%',
                    xy=(lambda_values[i], overflow_percentages[i]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontweight='bold')
    
    ax.set_title("Overflow Percentage vs Lambda", fontsize=16, fontweight='bold')
    ax.set_xlabel("Lambda (λ)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Overflow (%)", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right')
    
    if lambda_values:
        ax.set_xlim(0, max(lambda_values) * 1.05)
    if overflow_percentages:
        ax.set_ylim(0, max(overflow_percentages) * 1.1)
    
    filename = os.path.join(save_folder, "overflow_vs_lambda.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Graph saved as: {filename}")

# ----------------------------------------------------------------------------
# Gráfica 3: Gráfico 3D de MIT/IT ratio vs lambda vs overflow
# ----------------------------------------------------------------------------
def run_3d_graph(config):
    from scenario import run_simulation

    orig_dur    = config["simulation"]["duration"]
    orig_mit    = config["simulation"]["MIT"]
    orig_lambda = config["event_generator"]["lambda"]
    it_base     = config["simulation"]["IT"]

    lambda_values = np.arange(0.05, 0.5 + 1e-9, 0.05)
    mit_it_ratios = np.arange(0.5, 5.0 + 1e-9, 0.5)

    config["simulation"]["duration"] = min(orig_dur, 200)

    results = []
    for lam in lambda_values:
        for ratio in mit_it_ratios:
            MIT_test = it_base * ratio
            config["event_generator"]["lambda"] = lam
            config["simulation"]["MIT"]         = MIT_test

            _, _, stats = run_simulation(config)

            results.append({
                "Lambda":       lam,
                "MIT_IT_Ratio": ratio,
                "Overflow":     stats["overflow_percentage"]
            })
            print(f"λ={lam:.2f}, ratio={ratio:.2f} → MIT={MIT_test:.2f}, overflow={stats['overflow_percentage']:.1f}%")

    config["simulation"]["duration"]    = orig_dur
    config["event_generator"]["lambda"] = orig_lambda
    config["simulation"]["MIT"]         = orig_mit

    df = pd.DataFrame(results)
    save_folder = get_save_folder(config)
    csv_path = os.path.join(save_folder, "3d_graph_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data CSV guardado en: {csv_path}")

    pivot = (df.pivot(index="MIT_IT_Ratio", columns="Lambda", values="Overflow")
               .sort_index(axis=0).sort_index(axis=1))
    X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
    Z    = pivot.values

    fig = plt.figure(figsize=(12,10))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.scatter(X.ravel(), Y.ravel(), Z.ravel(),
               c=Z.ravel(), cmap="viridis", edgecolor="k", s=30)

    ax.set_xlabel("Lambda (λ)")
    ax.set_ylabel("MIT/IT Ratio")
    ax.set_zlabel("Overflow (%)")
    ax.set_title("Overflow vs λ (0.05–0.5) y MIT/IT (0.5–5.0)")
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12)

    # Ajuste de la cámara para una mejor inclinación y que el eje Z quede a la derecha
    ax.view_init(elev=20, azim=150)

    out_png = os.path.join(save_folder, "3d_overflow_surface.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Superficie 3D guardada en: {out_png}")
