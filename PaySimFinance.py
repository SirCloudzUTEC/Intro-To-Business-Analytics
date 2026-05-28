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
print("1.3 Adelantado.\n    ")
print("Debido a que nuestro dataset es bastante grande, vamos a trabajar con una muestra reducida del mismo. \nEsto nos permitirá realizar análisis y visualizaciones de manera más eficiente sin perder la esencia de los datos.")
dfOriginal = pd.read_csv('paysim.csv')
print(f"El dataset original tiene {dfOriginal.shape[0]} filas y {dfOriginal.shape[1]} columnas.")

df = dfOriginal.sample(n=100_000, random_state=42).reset_index(drop=True)
print(f"El dataset reducido tiene {df.shape[0]} filas y {df.shape[1]} columnas.")
#df.to_csv('PaySim_Reducido.csv', index=False)

salto()

# Estudio puro y duro del dataset ya reducido
print("1.1\n")
print("Muestra de las primeras 5 registros del dataset:")
print(df.head(5))
print("""
Descripción de columnas (contexto de negocio):
  step          → Hora de simulación (1 step ≈ 1 hora). Permite ordenar eventos en el tiempo.
  type          → Tipo de transacción: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER.
  amount        → Monto de la transacción en moneda local.
  nameOrig      → ID del cliente que origina la transacción.
  oldbalanceOrg → Saldo del originador ANTES de la transacción.
  newbalanceOrig→ Saldo del originador DESPUÉS de la transacción.
  nameDest      → ID del destinatario de la transacción.
  oldbalanceDest→ Saldo del destinatario ANTES de la transacción.
  newbalanceDest→ Saldo del destinatario DESPUÉS de la transacción.
  isFraud       → Variable objetivo: 1 = transacción fraudulenta, 0 = legítima.
  isFlaggedFraud→ Marcas internas del sistema antifraude.
""")
salto()

print("1.2\n")
print('=== MEMORY USAGE ===')
print(f'Uso de memoria total en dataset sin reducir: {dfOriginal.memory_usage(deep=True).sum()} bytes')
print()
print(f'Uso de memoria total en dataset reducido: {df.memory_usage(deep=True).sum()} bytes\n')

print('=== TIPOS DE DATOS ===')
print(df.dtypes)
print()

print("1.4\nQueremos responder preguntas de negocio de si una transaccion es fraudulenta o no.\nLa variable mas importante en ese caso es: 'isFraud'")

salto()

# Info acerca de nulos
print("2.1: Tabla de valores nulos en el dataset:")
nulos = pd.DataFrame({
    'Nulos':df.isnull().sum(),
    'Porcentaje': (df.isnull().sum()/len(df) * 100).round(2)
}).query('Nulos > 0').sort_values(by='Nulos', ascending=False)
if nulos.empty:
    print("No existen valores nulos en el dataset. No tenemos que hacer limpieza en este aspecto.")
else:
    print(nulos)
salto()


print("2.2 Detección de duplicados")

duplicados = df.duplicated().sum()
porcentaje_duplicados = (duplicados / len(df)) * 100

print(f"Cantidad de duplicados exactos: {duplicados}")
print(f"Porcentaje de duplicados: {porcentaje_duplicados:.4f}%")

if duplicados > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Nuevo tamaño del dataset luego de eliminar duplicados: {df.shape}")
else:
    print("No existen filas duplicadas exactas.")

salto()


print("2.3 Verificación de tipos de datos")

print("Tipos actuales:")
print(df.dtypes)

# Conversion de variables categóricas
df['type'] = df['type'].astype('category')

# Conversion de variables binarias
df['isFraud'] = df['isFraud'].astype('int8')
df['isFlaggedFraud'] = df['isFlaggedFraud'].astype('int8')

# Conversion columnas string a category
df['nameOrig'] = df['nameOrig'].astype('category')
df['nameDest'] = df['nameDest'].astype('category')

print("\nTipos luego de conversiones:")
print(df.dtypes)

print("""
Documentación de conversiones realizadas:
- type -> category
- nameOrig -> category
- nameDest -> category
- isFraud -> int8
- isFlaggedFraud -> int8

No se encontraron fechas almacenadas como texto.
No se encontraron variables numéricas almacenadas como texto.
""")

salto()



print("2.4 Detección de inconsistencias")

# --- Montos negativos ---
montos_negativos = df[df['amount'] < 0]

# --- Balances negativos ---
balances_negativos = df[
    (df['oldbalanceOrg'] < 0) |
    (df['newbalanceOrig'] < 0) |
    (df['oldbalanceDest'] < 0) |
    (df['newbalanceDest'] < 0)
]

# --- Balances inconsistentes ---
# Ejemplo:
# si amount > oldbalanceOrg y aun así la transacción ocurre
balances_inconsistentes = df[
    (df['type'].isin(['TRANSFER', 'CASH_OUT'])) &
    (df['amount'] > df['oldbalanceOrg']) &
    (df['oldbalanceOrg'] > 0)
]

print(f"Registros con montos negativos: {len(montos_negativos)}")
print(f"Registros con balances negativos: {len(balances_negativos)}")
print(f"Registros con balances inconsistentes: {len(balances_inconsistentes)}")

print("""Decisión:\n
NO hay montos ni balances negativos, por lo cual no es necesario hacer eliminacion de filas.\n
Los balances inconsistentes se mantendrán porque podrían representar
anomalías reales o comportamientos asociados a fraude.""")

# Eliminación de inconsistencias imposibles
df = df[
    (df['amount'] >= 0) &
    (df['oldbalanceOrg'] >= 0) &
    (df['newbalanceOrig'] >= 0) &
    (df['oldbalanceDest'] >= 0) &
    (df['newbalanceDest'] >= 0)
].reset_index(drop=True)

print("\nLimpieza completada.")
print(f"Tamaño del dataset se mantiene: {df.shape}")

salto()



print("2.5 Tabla resumen de calidad")
    
resumen_calidad = pd.DataFrame({
    'Variable': df.columns,
    'Tipo': [str(df[col].dtype) for col in df.columns],
    'Nulos (%)': [
        round((df[col].isnull().sum() / len(df)) * 100, 2)
        for col in df.columns
    ],
    'Duplicados': [
        duplicados if i == 0 else "-"
        for i in range(len(df.columns))
    ],
    'Inconsistencias': [
        (
            len(df[df[col] < 0])
            if str(df[col].dtype) != 'category'
            and col not in ['isFraud', 'isFlaggedFraud', 'step']
            else "-"
        )
        for col in df.columns
    ],
    'Acción tomada': [
        'Conversión de tipos y validación'
        for _ in df.columns
    ]
})

display(resumen_calidad)

print("\nTabla de calidad generada correctamente.")

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



#ACTIVIDAD 3

# =========================================================
# ===== EDA NUMÉRICO UNIVARIADO ===========================
# =========================================================

print("=== ANÁLISIS UNIVARIADO DE VARIABLES NUMÉRICAS ===")
print()

# Amount 
print()
print("=== ANÁLISIS DE AMOUNT ===")

feature = 'amount'
print(df[feature].describe())
print()
print(f"Asimetría: {df[feature].skew():.2f}")
print(f"Curtosis: {df[feature].kurtosis():.2f}")

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x=feature,
    kde=True,
    color='steelblue',
    edgecolor='white',
    linewidth=0.5
)
#plt.xlim(0, 2)
#plt.ylim(0, 20000)

mean_val = df[feature].mean()
median_val = df[feature].median()

plt.axvline(
    mean_val,
    color='red',
    linestyle='--',
    linewidth=1.5,
    label=f'Media: {mean_val:.2f}'
)

plt.axvline(
    median_val,
    color='orange',
    linestyle='-',
    linewidth=1.5,
    label=f'Mediana: {median_val:.2f}'
)

plt.title(f'Distribución de {feature}')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.legend()
plt.tight_layout()

plt.show()
plt.close()

#OldbalanceOrg 
print()
print("=== ANÁLISIS DE oldbalanceOrg ===")

feature = 'oldbalanceOrg'
print(df[feature].describe())
print()
print(f"Asimetría: {df[feature].skew():.2f}")
print(f"Curtosis: {df[feature].kurtosis():.2f}")

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x=feature,
    kde=True,
    color='steelblue',
    edgecolor='white',
    linewidth=0.5
)
#plt.xlim(0, 2)
#plt.ylim(0, 20000)

mean_val = df[feature].mean()
median_val = df[feature].median()

plt.axvline(
    mean_val,
    color='red',
    linestyle='--',
    linewidth=1.5,
    label=f'Media: {mean_val:.2f}'
)

plt.axvline(
    median_val,
    color='orange',
    linestyle='-',
    linewidth=1.5,
    label=f'Mediana: {median_val:.2f}'
)

plt.title(f'Distribución de {feature}')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.legend()
plt.tight_layout()

plt.show()
plt.close()

#newbalanceOrig 
print()
print("=== ANÁLISIS DE newbalanceOrig ===")

feature = 'newbalanceOrig'
print(df[feature].describe())
print()
print(f"Asimetría: {df[feature].skew():.2f}")
print(f"Curtosis: {df[feature].kurtosis():.2f}")

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x=feature,
    kde=True,
    color='steelblue',
    edgecolor='white',
    linewidth=0.5
)
#plt.xlim(0, 2)
#plt.ylim(0, 20000)

mean_val = df[feature].mean()
median_val = df[feature].median()

plt.axvline(
    mean_val,
    color='red',
    linestyle='--',
    linewidth=1.5,
    label=f'Media: {mean_val:.2f}'
)

plt.axvline(
    median_val,
    color='orange',
    linestyle='-',
    linewidth=1.5,
    label=f'Mediana: {median_val:.2f}'
)

plt.title(f'Distribución de {feature}')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.legend()
plt.tight_layout()

plt.show()
plt.close()

#oldbalanceDest 
print()
print("=== ANÁLISIS DE oldbalanceDest ===")

feature = 'oldbalanceDest'
print(df[feature].describe())
print()
print(f"Asimetría: {df[feature].skew():.2f}")
print(f"Curtosis: {df[feature].kurtosis():.2f}")

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x=feature,
    kde=True,
    color='steelblue',
    edgecolor='white',
    linewidth=0.5
)
#plt.xlim(0, 2)
#plt.ylim(0, 20000)

mean_val = df[feature].mean()
median_val = df[feature].median()

plt.axvline(
    mean_val,
    color='red',
    linestyle='--',
    linewidth=1.5,
    label=f'Media: {mean_val:.2f}'
)

plt.axvline(
    median_val,
    color='orange',
    linestyle='-',
    linewidth=1.5,
    label=f'Mediana: {median_val:.2f}'
)

plt.title(f'Distribución de {feature}')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()



#newbalanceDest 
print()
print("=== ANÁLISIS DE newbalanceDest ===")

feature = 'newbalanceDest'
print(df[feature].describe())
print()
print(f"Asimetría: {df[feature].skew():.2f}")
print(f"Curtosis: {df[feature].kurtosis():.2f}")

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x=feature,
    kde=True,
    color='steelblue',
    edgecolor='white',
    linewidth=0.5
)
#plt.xlim(0, 2)
#plt.ylim(0, 20000)

mean_val = df[feature].mean()
median_val = df[feature].median()

plt.axvline(
    mean_val,
    color='red',
    linestyle='--',
    linewidth=1.5,
    label=f'Media: {mean_val:.2f}'
)

plt.axvline(
    median_val,
    color='orange',
    linestyle='-',
    linewidth=1.5,
    label=f'Mediana: {median_val:.2f}'
)

plt.title(f'Distribución de {feature}')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.legend()
plt.tight_layout()

plt.show()
plt.close()


salto()





"""
# =========================================================
# ===== EDA NUMÉRICO UNIVARIADO ===========================
# =========================================================

print("=== ANÁLISIS UNIVARIADO DE VARIABLES NUMÉRICAS ===")

variables_numericas = [
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest'
]

for col in variables_numericas:
    
    print(f"\n--- {col.upper()} ---")
    
    print(df[col].describe())
    
    print(f"Asimetría (Skewness): {df[col].skew():.2f}")
    print(f"Curtosis: {df[col].kurtosis():.2f}")
    
    # Histograma
    fig, ax = plt.subplots()

    sns.histplot(
        df[col],
        bins=50,
        kde=True,
        ax=ax,
        color='#3498db'
    )

    # Media y mediana
    mean_val = df[feature].mean()
    median_val = df[feature].median()

    axes[i].axvline(
        mean_val,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label=f'Media: {mean_val:.2f}'
    )

    axes[i].axvline(
        median_val,
        color='orange',
        linestyle='-',
        linewidth=1.5,
        label=f'Mediana: {median_val:.2f}'
    )

    axes[i].set_title(
        feature.replace('_', ' ').title(),
        fontsize=11,
        fontweight='bold'
    )

    axes[i].set_xlabel('Valor')
    axes[i].set_ylabel('Frecuencia')

    axes[i].legend(fontsize=8)

    plt.suptitle(
        'Distribución de Variables Numéricas',
        fontsize=16,
        fontweight='bold',
        y=1.02)

    plt.tight_layout()
    plt.show()

salto()

"""



# Actividad 4.1 

print("=== ANÁLISIS UNIVARIADO DE VARIABLES NUMÉRICAS ===")

variables_numericas = [
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest'
]

for col in variables_numericas:
    
    print(f"\n--- {col.upper()} ---")

    fig, ax = plt.subplots(figsize=(10,5))

    sns.histplot(
        data=df,
        x=col,
        bins=50,
        kde=True,
        ax=ax,
        color='#3498db',
        edgecolor='white'
    )

    # Media y mediana
    mean_val = df[col].mean()
    median_val = df[col].median()

    ax.axvline(
        mean_val,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label=f'Media: {mean_val:.2f}'
    )

    ax.axvline(
        median_val,
        color='orange',
        linestyle='-',
        linewidth=1.5,
        label=f'Mediana: {median_val:.2f}'
    )

    ax.set_title(f'Distribución de {col}', fontsize=12)
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel('Frecuencia', fontsize=10)

    ax.legend()

    plt.tight_layout()
    plt.show()

    salto()