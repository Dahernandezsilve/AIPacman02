import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
data = pd.read_csv('./pacman_results_f.csv')

# Eliminar las filas donde se usa mediumClassic como layout
data = data[data['Layout'] != 'mediumClassic']

# Limpieza de datos (si es necesario)
# Asumimos que los datos están limpios para este ejemplo

# Análisis estadístico básico
print(data.describe())

# Visualización

# Distribución de puntuaciones por agente y profundidad
plt.figure(figsize=(12, 6))
sns.boxplot(x='Depth', y='Score', hue='Agent', data=data)
plt.title('Distribución de puntuaciones por agente y profundidad')
plt.xlabel('Profundidad')
plt.ylabel('Puntuación')
plt.legend(title='Agente')
plt.show()

# Proporciones de victorias/derrotas
win_loss = data.groupby(['Agent', 'Depth', 'Result']
                        ).size().unstack().fillna(0)
win_loss_ratio = win_loss.div(win_loss.sum(axis=1), axis=0)
win_loss_ratio.plot(kind='barh', stacked=True, figsize=(12, 8))
plt.title('Proporciones de victorias/derrotas por agente y profundidad')
plt.xlabel('Agente y profundidad')
plt.ylabel('Porcentaje')

# Ajustar el margen inferior
plt.subplots_adjust(left=0.2)

plt.show()

# Análisis del tiempo de ejecución
plt.figure(figsize=(12, 6))
sns.lineplot(x='Depth', y='Execution Time', hue='Agent', data=data, marker='o')
plt.title('Tiempo de ejecución por profundidad y agente')
plt.xlabel('Profundidad')
plt.ylabel('Tiempo de ejecución (segundos)')
plt.legend(title='Agente')
plt.grid(True)
plt.show()
