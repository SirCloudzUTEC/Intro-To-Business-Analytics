import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from IPython.display import display, Markdown
import os

#Configs
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
sns.set_palette('Set2')


#Funciones de orden y limpieza de terminal
def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')

limpiar_pantalla()

def salto():
    print("\n" + "="*65 + "\n")
    

print("Configuraciones hechas.")
salto()


#Explicacion de la reduccion del dataset
print("Debido a que nuestro dataset es bastante grande, vamos a trabajar con una muestra reducida del mismo. \nEsto nos permitirá realizar análisis y visualizaciones de manera más eficiente sin perder la esencia de los datos.")
dfOriginal = pd.read_csv('paysim.csv')
print(f"El dataset original tiene {dfOriginal.shape[0]} filas y {dfOriginal.shape[1]} columnas.")

df = dfOriginal.sample(n=100_000, random_state=42).reset_index(drop=True)
print(f"El dataset reducido tiene {df.shape[0]} filas y {df.shape[1]} columnas.")
df.to_csv('PaySim_Reducido.csv', index=False)

salto()

# Estudio puro y duro del dataset ya reducido
print("Muestra de los primeros 6 registros del dataset:")
print(df.head(6))
salto()

print('=== TIPOS DE DATOS ===')
print(df.dtypes)
print()
print('=== DESCRIPCIÓN ESTADÍSTICA ===')
print(df.describe())
salto()


# Info acerca de nulos
print("Tabla de valores nulos en el dataset:")
nulos = pd.DataFrame({
    'Nulos':df.isnull().sum(),
    'Porcentaje': (df.isnull().sum()/len(df) * 100).round(2)
}).query('Nulos > 0').sort_values(by='Nulos', ascending=False)
if nulos.empty:
    print("No existen valores nulos en el dataset. No tenemos que hacer limpieza en este aspecto.")
else:
    print(nulos)
salto()


# ===== EDA =====

print("EDA Inicial")

fig, ax = plt.subplots()

df['isFraud'].value_counts().plot(
    kind='bar',
    ax=ax,
    color=['#2ecc71', '#e74c3c'],
    edgecolor='white'
)

ax.set_title(
    'Distribución de Transacciones Fraudulentas vs No Fraudulentas',
    fontsize=12
)

ax.set_xlabel('Es fraude? \n (0=NO, 1=SI)', fontsize=10)
ax.set_ylabel('Cantidad', fontsize=10)

for i, v in enumerate(df['isFraud'].value_counts()):
    ax.text(i, v + 5, f'{v}', ha='center', fontweight='bold')

plt.show()
print("Grafico entregado.")
print("Podemos observar que la mayoría de las transacciones no son fraudulentas")
print("\nEncontramos:\n99859 Transacciones No Fraudulentas\n141 Transacciones Fraudulentas")
# Aqui el objetivo proximo es guardar la imagen. Tarea proxima

# Metodos de pago
fig, ax = plt.subplots()

df['type'].value_counts().sort_index().plot(
    kind='bar',
    ax=ax,
    color=['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#f1c40f'],
    edgecolor='white'
)

ax.set_title('Distribución de Tipos de Transacción', fontsize=12)
ax.set_xlabel('Tipo de Transacción', fontsize=10)
ax.set_ylabel('Cantidad', fontsize=10)

for i, v in enumerate(df['type'].value_counts().sort_index()):
    ax.text(i, v + 5, f'{v}', ha='center', fontweight='bold')

plt.show()
print("Grafico entregado.")

salto()


