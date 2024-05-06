import subprocess
import time
import matplotlib.pyplot as plt

# Vaciar el contenido del archivo de registro al inicio del programa
with open('scores.txt', 'w') as f:
    f.write('')

# Número de veces que deseas ejecutar el comando
num_runs = 20

# Listas para almacenar los tiempos de ejecución y los puntajes de victorias
execution_times = []
victory_scores = []

# Ejecutar el comando varias veces y recopilar tiempos de ejecución y puntajes de victorias
for i in range(num_runs):
    start_time = time.time()  # Registrar el tiempo de inicio de la ejecución
    process = subprocess.Popen(["python", "pacman.py", "--frameTime", "0", "-p", "ReflexAgent", "-k", "2"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.communicate()  # Esperar a que el proceso se complete
    end_time = time.time()  # Registrar el tiempo de finalización de la ejecución
    execution_time = end_time - start_time  # Calcular el tiempo de ejecución
    execution_times.append(execution_time)  # Agregar el tiempo de ejecución a la lista

    # Leer el puntaje del último juego
    with open('scores.txt', 'r') as f:
        last_line = f.readlines()[-1].strip()
        parts = last_line.split(", ")
        score = float(parts[0].split(": ")[1])  # Extraer el puntaje numérico
        result = parts[1].split(": ")[1]
        if result == "win":
            victory_scores.append(score)  # Agregar el puntaje a la lista

# Calcular el tiempo promedio de ejecución
average_execution_time = sum(execution_times) / num_runs

# Leer el archivo de registro de puntajes y resultados
with open('scores.txt', 'r') as f:
    lines = f.readlines()

victories = 0
defeats = 0
results = []

# Contar victorias y derrotas, y recopilar resultados
for line in lines:
    if "win" in line:
        victories += 1
        results.append('win')
    elif "lose" in line:
        defeats += 1
        results.append('lose')

# Generar la gráfica comparativa de victorias y derrotas
plt.figure(figsize=(16, 6))

plt.subplot(1, 3, 1)
labels = ['Victorias', 'Derrotas']
sizes = [victories, defeats]
colors = ['lightgreen', 'lightcoral']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Comparativa de Victorias y Derrotas')

# Generar la gráfica del tiempo de ejecución
plt.subplot(1, 3, 2)
for i, result in enumerate(results):
    color = 'green' if result == 'win' else 'red'
    plt.plot([i + 1, i + 1], [0, execution_times[i]], color=color, linewidth=6)

plt.title('Tiempo de ejecución vs. Número de ejecuciones')
plt.xlabel('Número de ejecuciones')
plt.ylabel('Tiempo de ejecución (segundos)')
plt.grid(True)

# Generar la gráfica de puntajes de victorias
plt.subplot(1, 3, 3)
plt.plot(range(1, len(victory_scores) + 1), victory_scores, marker='o', linestyle='-')
plt.title('Puntajes de Victorias')
plt.xlabel('Número de victorias')
plt.ylabel('Puntaje')
plt.grid(True)

plt.tight_layout()
plt.show()
