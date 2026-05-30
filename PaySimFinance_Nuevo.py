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
dfOriginal = pd.read_csv('PaySim_Reducido.csv')
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


# 3.1 y 3.2 para copiar


print("3.3 Análisis de variables categóricas\n")

# Seleccionar variables categóricas
columnas_categoricas = df.select_dtypes(include=['category', 'object']).columns

for col in columnas_categoricas:

    print("=" * 70)
    print(f"Variable categórica: {col}\n")

    # Cantidad de categorías únicas
    cantidad_categorias = df[col].nunique()

    print(f"Número de categorías únicas: {cantidad_categorias}\n")

    # Frecuencia absoluta
    print("=== Frecuencia Absoluta ===")
    frecuencia_absoluta = df[col].value_counts()
    print(frecuencia_absoluta)

    print()

    # Frecuencia relativa
    print("=== Frecuencia Relativa (%) ===")
    frecuencia_relativa = (
        df[col]
        .value_counts(normalize=True)
        .mul(100)
        .round(4)
    )

    print(frecuencia_relativa)

    print()

    # Categorías con menos del 1%
    categorias_menores_1 = frecuencia_relativa[frecuencia_relativa < 1]

    if len(categorias_menores_1) > 0:
        print("Categorías con menos del 1% de representación:\n")
        print(categorias_menores_1)
    else:
        print("No existen categorías con menos del 1% de representación.")

    print("\n")
salto()




print("3.4 Variable Objetivo Balanceada?")
ig, ax = plt.subplots()

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
print("Grafico entregado.\n")
print("Podemos observar que la mayoría de las transacciones no son fraudulentas")
print("\nEncontramos:\n99859 Transacciones No Fraudulentas\n141 Transacciones Fraudulentas")
print("Existe un claro desbalance en la variable objetivo. El gran problema viene al tratar de crear un modelo predictivo, el cual podría tender a predecir siempre la clase mayoritaria (no fraude) y aun así obtener una alta precisión, pero sin detectar correctamente los casos de fraude.")
salto()


print("3.4 Variables con mayor variabilidad\n")

# Seleccionar solo variables numéricas
df_numerico = df.select_dtypes(include=[np.number])

# Calcular coeficiente de variación
cv = (df_numerico.std() / df_numerico.mean()).sort_values(ascending=False)

# Top 3 variables con mayor variabilidad
top3 = cv.head(3)

print("=== Top 3 Variables con Mayor Variabilidad ===")
print(top3)
print()

# ===== Gráfico =====

fig, ax = plt.subplots(figsize=(10, 5))

top3.plot(
    kind='bar',
    ax=ax,
    color=['#3498db', '#9b59b6', '#1abc9c'],
    edgecolor='white'
)

ax.set_title(
    'Top 3 Variables con Mayor Variabilidad',
    fontsize=13
)

ax.set_xlabel('Variables', fontsize=10)
ax.set_ylabel('Coeficiente de Variación', fontsize=10)

# Etiquetas sobre las barras
for i, v in enumerate(top3):
    ax.text(
        i,
        v + 0.1,
        f'{v:.2f}',
        ha='center',
        fontweight='bold'
    )

plt.xticks(rotation=0)
plt.show()

salto()




# ===== EDA =====

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





# ============================================================
# Actividad 4.1 

print("4.1. Analisis univariado de vairables numericas")

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



# ============================================================
# ACTIVIDAD 4.3 

print("4.3 Transformación logarítmica para variables con skewness > 2")

variables_numericas = [
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest'
]

skewness_vals = df[variables_numericas].skew()

vars_a_transformar = skewness_vals[skewness_vals > 2].index.tolist()

print()
print("=== Skewness por variable ===")
print(skewness_vals.round(4))
print()
print()

for col in vars_a_transformar:

    print(f"--- Transformación aplicada a: {col.upper()} ---")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(
        data=df,
        x=col,
        bins=50,
        kde=True,
        ax=axes[0],
        color='#3498db',
        edgecolor='white'
    )
    axes[0].axvline(df[col].mean(), color='red', linestyle='--',linewidth=1.5, label=f'Media: {df[col].mean():.2f}')
    axes[0].axvline(df[col].median(), color='orange',linestyle='-', linewidth=1.5, label=f'Mediana: {df[col].median():.2f}')
    axes[0].set_title(f'{col} — ANTES (skewness: {df[col].skew():.2f})', fontsize=11)
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Frecuencia')
    axes[0].legend()

    #Transformación
    col_log = np.log1p(df[col])

    sns.histplot(
        x=col_log,
        bins=50,
        kde=True,
        ax=axes[1],
        color='#1abc9c',
        edgecolor='white'
    )
    axes[1].axvline(col_log.mean(),   color='red',    linestyle='--', linewidth=1.5, label=f'Media: {col_log.mean():.2f}')
    axes[1].axvline(col_log.median(), color='orange', linestyle='-',  linewidth=1.5, label=f'Mediana: {col_log.median():.2f}')
    axes[1].set_title(f'log1p({col}) — DESPUÉS (skewness: {col_log.skew():.2f})', fontsize=11)
    axes[1].set_xlabel(f'log1p({col})')
    axes[1].set_ylabel('Frecuencia')
    axes[1].legend()

    plt.suptitle(f'Comparación Antes / Después — {col}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Confirmación textual de mejora
    skew_antes  = df[col].skew()
    skew_despues = col_log.skew()

    if abs(skew_despues) < abs(skew_antes):
        print()
        print(f"La transformación log1p mejoró la simetría de '{col}'.")
        print(f"Skewness antes: {skew_antes:.4f}")
        print(f"Skewness después: {skew_despues:.4f}\n")
    else:
        print(f"✘ La transformación log1p NO mejoró la simetría de '{col}'.")
        print(f"Skewness antes: {skew_antes:.4f}")
        print(f"Skewness después: {skew_despues:.4f}\n")

salto()


# ============================================================
# ACTIVIDAD 4.5 

print("4.5 Distribución de amount separada por variable objetivo (isFraud)\n")

df['amount_log'] = np.log1p(df['amount'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#Gráfico izquierdo: escala original 
sns.histplot(
    data=df,
    x='amount',
    hue='isFraud',
    bins=60,
    kde=True,
    stat='density',
    common_norm=False,
    ax=axes[0],
    palette={0: '#2ecc71', 1: '#e74c3c'},
    alpha=0.5,
    edgecolor='white'
)
axes[0].set_title('Distribución de amount por isFraud\n(densidad)', fontsize=11)
axes[0].set_xlabel('Monto de la transacción')
axes[0].set_ylabel('Densidad')

#Gráfico derecho: escala log1p
sns.histplot(
    data=df,
    x='amount_log',
    hue='isFraud',
    bins=60,
    kde=True,
    stat='density',
    common_norm=False,
    ax=axes[1],
    palette={0: '#2ecc71', 1: '#e74c3c'},
    alpha=0.5,
    edgecolor='white'
)
axes[1].set_title('Distribución de log1p(amount) por isFraud\n(escala logarítmica)', fontsize=11)
axes[1].set_xlabel('log1p(amount)')
axes[1].set_ylabel('Densidad')

plt.suptitle('Monto de transacción vs. Fraude', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("Gráfico entregado.")
salto()