import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------------------------------------------------------
# Función para obtener (y crear si no existe) el folder de guardado
# ----------------------------------------------------------------------------
def get_save_folder(config, mit=None, it=None, tunit=None):
    """Obtiene carpeta con valores actuales, no solo los del config."""
    mit = mit if mit is not None else config["simulation"]["MIT"]
    it = it if it is not None else config["simulation"]["IT"]
    tunit = tunit if tunit is not None else config["simulation"]["Tunit"]
    
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
    mit_value = config["simulation"]["MIT"]
    it_value = config["simulation"]["IT"]
    
    regular_series = get_series(processor.regular_monitor, duration)
    offloading_series = get_series(processor.offloading_monitor, duration)
    t = range(duration + 1)
    
    y_max = max(max(regular_series, default=0), max(offloading_series, default=0))
    y_max = max(20, int(y_max * 1.2))
    
    fig, (ax_off, ax_reg) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Window Occupancy (λ={lambda_value}, MIT={mit_value}, IT={it_value})", fontsize=16)
    
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
# Gráfica 2: Barrido de Lambda - CORREGIDA con anotaciones
# ----------------------------------------------------------------------------
def run_lambda_sweep(config):
    from scenario3 import run_simulation

    lambda_values = [0.001, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                     0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    overflow_percentages = []
    results_data = []
    original_lambda = config["event_generator"]["lambda"]

    print("\nRunning lambda sweep simulation...")
    print(f"Using fixed MIT={config['simulation']['MIT']}, IT={config['simulation']['IT']}")
    
    for lam in lambda_values:
        config["event_generator"]["lambda"] = lam
        _, processor, stats = run_simulation(config)
        
        overflow_percentages.append(stats['overflow_percentage'])
        results_data.append({
            'Lambda': lam,
            'Overflow (%)': stats['overflow_percentage'],
            'Events Generated': stats['events_generated'],
            'Events Processed': stats['events_processed'],
            'Events Offloaded': stats['events_offloaded']
        })
        print(f"Lambda: {lam:.3f} → Overflow={stats['overflow_percentage']:.2f}%")

    # Restaurar lambda
    config["event_generator"]["lambda"] = original_lambda

    # Guardar CSV
    save_folder = get_save_folder(config)
    csv_file = os.path.join(save_folder, "lambda_sweep_results.csv")
    pd.DataFrame(results_data).to_csv(csv_file, index=False)

    # Graficar
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Línea principal
    ax.plot(lambda_values, overflow_percentages, marker='o', linestyle='-',
            color='green', linewidth=2, markersize=8, label='Overflow %')

    # Línea de tendencia polinómica
    z = np.polyfit(lambda_values, overflow_percentages, 3)
    p = np.poly1d(z)
    x_smooth = np.linspace(min(lambda_values), max(lambda_values), 100)
    ax.plot(x_smooth, p(x_smooth), 'r--', linewidth=1, alpha=0.7,
            label='Trend (degree 3)')

    # Ecuación en la gráfica
    eq = f"y = {z[0]:.3f}x³ + {z[1]:.3f}x² + {z[2]:.3f}x + {z[3]:.3f}"
    ax.text(0.05, 0.95, eq, transform=ax.transAxes, fontsize=11,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    # Anotar punto final
    final_overflow = overflow_percentages[-1]
    ax.annotate(f'{final_overflow:.1f}%',
                xy=(1.0, final_overflow),
                xytext=(0.95, final_overflow + 3),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.6),
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    # Anotar algunos puntos intermedios clave
    for i in [0, len(lambda_values)//2]:
        ax.annotate(f'{overflow_percentages[i]:.1f}%',
                    xy=(lambda_values[i], overflow_percentages[i]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_title(f"Overflow Percentage vs Lambda (MIT={config['simulation']['MIT']}, IT={config['simulation']['IT']})", 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel("Lambda (λ)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Overflow (%)", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right', fontsize=12)
    ax.set_xlim(0, max(lambda_values)*1.05)
    ax.set_ylim(0, max(overflow_percentages)*1.1)

    plt.tight_layout()
    filename = os.path.join(save_folder, "overflow_vs_lambda.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Graph saved as: {filename}")

# ----------------------------------------------------------------------------
# Gráfica 3: Gráfico 3D - CORREGIDA sin puntos rojos y con línea de config
# ----------------------------------------------------------------------------
def run_3d_graph(config):
    from scenario3 import run_simulation
    
    print("\n=== Generating 3D graph data ===")
    
    # Guardar valores originales
    original_duration = config["simulation"]["duration"]
    original_lambda = config["event_generator"]["lambda"]
    
    # Reducir duración para acelerar
    config["simulation"]["duration"] = 1000
    
    # Parámetros fijos y variables
    it_fijo = config["simulation"]["IT"]
    config_mit = config["simulation"]["MIT"]
    config_ratio = config_mit / it_fijo
    
    # Reducir rango de MIT/IT para que la gráfica no se vea plana
    lambda_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    mit_it_ratios = [1, 2, 3, 4, 5, 6]  # Rango más limitado
    
    print(f"Configuración actual: MIT={config_mit}, IT={it_fijo}, MIT/IT={config_ratio:.1f}")
    print(f"IT fijo: {it_fijo}")
    print(f"MIT variará de {it_fijo} a {it_fijo * max(mit_it_ratios)}")
    print(f"Lambda valores: {lambda_values}")
    print(f"Total simulaciones: {len(lambda_values) * len(mit_it_ratios)}")
    
    results = []
    
    for lam in lambda_values:
        config["event_generator"]["lambda"] = lam
        
        for ratio in mit_it_ratios:
            mit = it_fijo * ratio
            
            # Ejecutar simulación con parámetros específicos
            _, processor, stats = run_simulation(config, mit=mit, it=it_fijo)
            
            results.append({
                'Lambda': lam,
                'MIT_IT_Ratio': ratio,
                'MIT': stats['mit_actual'],
                'IT': stats['it_actual'],
                'Overflow': stats['overflow_percentage']
            })
            
            # Marcar si es la configuración actual
            is_config = (ratio == config_ratio and lam == original_lambda)
            marker = " ← CONFIG" if is_config else ""
            print(f"λ={lam:.2f}, MIT/IT={ratio:2d}, MIT={mit:.2f} → Overflow={stats['overflow_percentage']:.2f}%{marker}")
    
    # Restaurar valores
    config["simulation"]["duration"] = original_duration
    config["event_generator"]["lambda"] = original_lambda
    
    # Crear DataFrame y guardar
    results_df = pd.DataFrame(results)
    save_folder = get_save_folder(config)
    csv_file = os.path.join(save_folder, "3d_graph_data.csv")
    results_df.to_csv(csv_file, index=False)
    print(f"\nResultados guardados en: {csv_file}")
    
    # Verificar tendencias (corregir el problema con lambda=0.4)
    print("\n=== Verificación de Tendencias ===")
    for lam in lambda_values:
        datos_lam = results_df[results_df['Lambda'] == lam].sort_values('MIT_IT_Ratio')
        overflows = datos_lam['Overflow'].values
        
        # Verificar tendencia general (permitir pequeñas variaciones)
        if len(overflows) > 1:
            # Calcular tendencia general
            tendencia_general = overflows[-1] < overflows[0]  # Debería decrecer
            variaciones = sum(1 for i in range(len(overflows)-1) if overflows[i+1] > overflows[i])
            
            if tendencia_general and variaciones <= 1:  # Permitir máximo 1 inversión
                print(f"✓ Lambda={lam}: Overflow decrece con mayor MIT/IT (tendencia correcta)")
            else:
                print(f"⚠ Lambda={lam}: Anomalías en la tendencia ({variaciones} inversiones)")
    
    # Crear gráfica 3D
    pivot_table = results_df.pivot_table(
        index='MIT_IT_Ratio', 
        columns='Lambda', 
        values='Overflow'
    )
    
    X_vals = pivot_table.columns.values
    Y_vals = pivot_table.index.values
    X, Y = np.meshgrid(X_vals, Y_vals)
    Z = pivot_table.values
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Superficie principal
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                             rstride=1, cstride=1, antialiased=True,
                             linewidth=0, edgecolor='none')
    
    # Línea roja para marcar la configuración actual (MIT/IT = 5)
    if config_ratio in mit_it_ratios:
        # Obtener los valores de overflow para el ratio de configuración
        config_overflows = []
        for lam in lambda_values:
            overflow = results_df[(results_df['Lambda'] == lam) & 
                                (results_df['MIT_IT_Ratio'] == int(config_ratio))]['Overflow'].values
            if len(overflow) > 0:
                config_overflows.append(overflow[0])
            else:
                config_overflows.append(np.nan)
        
        # Dibujar línea roja para la configuración
        ax.plot(lambda_values, [config_ratio]*len(lambda_values), config_overflows,
                color='red', linewidth=4, label=f'Config: MIT/IT={config_ratio:.0f}',
                marker='o', markersize=8)
    
    # Configuración de ejes
    ax.set_xlabel('Lambda (λ)', fontsize=14, labelpad=10)
    ax.set_ylabel('MIT/IT Ratio', fontsize=14, labelpad=10)
    ax.set_zlabel('Overflow (%)', fontsize=14, labelpad=10)
    ax.set_title(f'3D Analysis: Overflow vs MIT/IT Ratio and Lambda\n' +
                 f'(IT={it_fijo}, Config: MIT={config_mit}, MIT/IT={config_ratio:.0f})', 
                 fontsize=16, pad=20)
    ax.view_init(elev=25, azim=45)
    
    # Barra de color
    cbar = fig.colorbar(surface, ax=ax, shrink=0.6, aspect=12)
    cbar.set_label('Overflow (%)', fontsize=12)
    
    # Contornos suaves en la base (sin picos)
    contours = ax.contour(X, Y, Z, zdir='z', offset=0, cmap='viridis', 
                          alpha=0.3, linewidths=1, levels=10)
    
    # Leyenda
    ax.legend(loc='upper right', fontsize=12)
    
    # Limitar ejes
    ax.set_xlim(0, max(lambda_values) * 1.05)
    ax.set_ylim(min(mit_it_ratios) * 0.9, max(mit_it_ratios) * 1.05)
    ax.set_zlim(0, min(100, max(results_df['Overflow']) * 1.1))
    
    plt.tight_layout()
    filename = os.path.join(save_folder, "3d_overflow_surface.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nGráfica 3D guardada como: {filename}")
    
    # Resumen
    print("\n=== Resumen ===")
    print(f"Overflow mínimo: {results_df['Overflow'].min():.2f}%")
    print(f"Overflow máximo: {results_df['Overflow'].max():.2f}%")
    print(f"\nConfiguración óptima:")
    optima = results_df.loc[results_df['Overflow'].idxmin()]
    print(f"Lambda={optima['Lambda']:.2f}, MIT/IT={optima['MIT_IT_Ratio']:.0f} → Overflow={optima['Overflow']:.2f}%")
    
    # Mostrar overflow para la configuración actual
    config_data = results_df[(results_df['Lambda'] == original_lambda) & 
                            (results_df['MIT_IT_Ratio'] == int(config_ratio))]
    if not config_data.empty:
        print(f"\nConfiguración actual (λ={original_lambda}, MIT/IT={config_ratio:.0f}):")
        print(f"Overflow = {config_data['Overflow'].values[0]:.2f}%")