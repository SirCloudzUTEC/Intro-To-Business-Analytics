import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False
sns.set_palette('Set2')

os.system('cls' if os.name == 'nt' else 'clear')

def salto():
    print("\n" + "=" * 65 + "\n")

def zscore(series):
    """Retorna la serie estandarizada con Z-Score."""
    return (series - series.mean()) / series.std()

print("Configuraciones hechas.")
salto()


# ── 1.3 — Justificación de la muestra (adelantado) ──────
print("1.3 — Justificación de la muestra\n")
print(
    "Debido a que el dataset original supera los 6 millones de registros, "
    "trabajar con él completo en análisis exploratorio sería innecesariamente "
    "costoso en tiempo y memoria.\nSe extrae una muestra aleatoria de 100,000 "
    "filas con random_state=42 para garantizar reproducibilidad.\n"
    "Una muestra de este tamaño es estadísticamente representativa: "
    "por la Ley de los Grandes Números, las distribuciones, proporciones "
    "y correlaciones observadas en la muestra convergen a las del universo."
)
salto()

df = pd.read_csv('PaySim_Reducido.csv')
print(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas.")
salto()


print("1.1 — Primeras 5 filas del dataset\n")
print(df.head(5).to_string())
print("""
Descripción de columnas (contexto de negocio):
  step           → Unidad de tiempo de la simulación (1 step ≈ 1 hora).
  type           → Tipo de transacción: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER.
  amount         → Monto de la transacción en moneda local (USD simulados).
  nameOrig       → ID del cliente que origina la transacción.
  oldbalanceOrg  → Saldo del originador ANTES de la transacción.
  newbalanceOrig → Saldo del originador DESPUÉS de la transacción.
  nameDest       → ID del destinatario.
  oldbalanceDest → Saldo del destinatario ANTES de la transacción.
  newbalanceDest → Saldo del destinatario DESPUÉS de la transacción.
  isFraud        → Variable objetivo: 1 = fraude, 0 = legítima.
  isFlaggedFraud → Flag interno del sistema antifraude del banco.
""")
salto()


print("1.2 — Dimensiones, memoria y tipos de dato\n")
filas, cols = df.shape
mem_bytes = df.memory_usage(deep=True).sum()
print(f"Filas    : {filas:,}")
print(f"Columnas : {cols}")
print(f"Memoria  : {mem_bytes:,} bytes  ({mem_bytes / 1024**2:.2f} MB)\n")
print("Tipos de dato por columna:")
print(df.dtypes.to_string())
salto()


print("1.4 — Pregunta de negocio\n")
print(
    "El dataset PaySim simula transacciones financieras móviles con el objetivo "
    "de estudiar el comportamiento del fraude en sistemas de pago digitales. "
    "La pregunta central que queremos responder es: ¿qué características de una "
    "transacción permiten identificarla como fraudulenta antes de que sea procesada? "
    "Para responderla, la variable más importante es 'isFraud', que actúa como "
    "etiqueta binaria de la clase objetivo en un eventual modelo de clasificación. "
    "Otras variables clave son 'amount' (el monto suele ser atípicamente alto en "
    "fraudes), 'type' (los fraudes se concentran en TRANSFER y CASH_OUT) y las "
    "diferencias de saldo antes/después de la transacción, que revelan patrones "
    "de vaciado de cuenta. Comprender estas relaciones es el primer paso para "
    "construir un sistema de detección de fraude confiable y accionable."
)
salto()


# ── 2.1 — Valores nulos ──────────────────────────────────
print("2.1 — Valores nulos\n")
nulos = pd.DataFrame({
    'Nulos':      df.isnull().sum(),
    'Porcentaje': (df.isnull().sum() / len(df) * 100).round(2)
}).sort_values('Nulos', ascending=False)

tiene_nulos = nulos[nulos['Nulos'] > 0]
if tiene_nulos.empty:
    print("No existen valores nulos en el dataset.")
    print("Estrategia de imputación: no aplica. El dataset está íntegro en este aspecto.")
else:
    print(tiene_nulos)
salto()


# ── 2.2 — Duplicados ─────────────────────────────────────
print("2.2 — Duplicados exactos\n")
n_dup = df.duplicated().sum()
pct_dup = (n_dup / len(df)) * 100
print(f"Duplicados encontrados : {n_dup}")
print(f"Porcentaje             : {pct_dup:.4f}%")

if n_dup > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dataset tras eliminación: {df.shape[0]:,} filas.")
else:
    print("No se requiere eliminación.")
salto()


# ── 2.3 — Conversión de tipos y eliminación de columnas irrelevantes ─────
print("2.3 — Verificación, conversión de tipos y eliminación de columnas\n")
print("Tipos originales:")
print(df.dtypes.to_string())

df['type']           = df['type'].astype('category')
df['isFraud']        = df['isFraud'].astype('int8')
df['isFlaggedFraud'] = df['isFlaggedFraud'].astype('int8')

# nameOrig / nameDest: IDs únicos de cliente, sin valor estadístico.
# step: unidad de tiempo de simulación; distribución aproximadamente uniforme
#       que no discrimina entre transacciones fraudulentas y legítimas.
cols_eliminar = ['nameOrig', 'nameDest', 'step']
df.drop(columns=cols_eliminar, inplace=True)

print("\nTipos tras conversión:")
print(df.dtypes.to_string())
print("""
Conversiones realizadas:
  type                      → category  (variable nominal; reduce uso de memoria)
  isFraud, isFlaggedFraud   → int8      (flags binarios; 1 byte es suficiente)

Columnas eliminadas:
  nameOrig, nameDest  → Identificadores únicos de cliente. No aportan valor
                        estadístico; solo generarían ruido en el análisis.
  step                → Unidad de tiempo de simulación. No discrimina fraude
                        y distrae del núcleo financiero del estudio.

No se encontraron fechas almacenadas como string.
No se encontraron variables numéricas almacenadas como object.
""")
salto()


# ── 2.4 — Valores imposibles / inconsistentes ────────────
print("2.4 — Detección de valores imposibles\n")

montos_neg = (df['amount'] < 0).sum()
bal_neg = (
    (df['oldbalanceOrg']  < 0) |
    (df['newbalanceOrig'] < 0) |
    (df['oldbalanceDest'] < 0) |
    (df['newbalanceDest'] < 0)
).sum()
bal_inc = df[
    (df['type'].isin(['TRANSFER', 'CASH_OUT'])) &
    (df['amount'] > df['oldbalanceOrg']) &
    (df['oldbalanceOrg'] > 0)
].shape[0]

print(f"Montos negativos          : {montos_neg}")
print(f"Balances negativos        : {bal_neg}")
print(f"Balances inconsistentes   : {bal_inc:,}  ({bal_inc/len(df)*100:.2f}% del total)")

print("""
Decisión:
  - Montos y balances negativos: 0 registros → no se requiere acción.
  - Balances inconsistentes (monto > saldo antes): se CONSERVAN.
    Justificación: en el contexto de fraude, que un retiro supere el saldo
    previo es una señal analítica valiosa; eliminar estos registros
    sesgaría el análisis hacia transacciones "normales".
""")
df = df[
    (df['amount']         >= 0) &
    (df['oldbalanceOrg']  >= 0) &
    (df['newbalanceOrig'] >= 0) &
    (df['oldbalanceDest'] >= 0) &
    (df['newbalanceDest'] >= 0)
].reset_index(drop=True)
print(f"Tamaño final del dataset: {df.shape[0]:,} filas.")
salto()


# ── 2.5 — Tabla resumen de calidad ──────────────────────
print("2.5 — Tabla resumen de calidad\n")

acciones = {
    'type':           'Convertido a category',
    'amount':         'Validado ≥ 0 (sin negativos)',
    'oldbalanceOrg':  'Validado ≥ 0 (sin negativos)',
    'newbalanceOrig': 'Validado ≥ 0 (sin negativos)',
    'oldbalanceDest': 'Validado ≥ 0 (sin negativos)',
    'newbalanceDest': 'Validado ≥ 0 (sin negativos)',
    'isFraud':        'Convertido a int8',
    'isFlaggedFraud': 'Convertido a int8',
}

resumen_cal = pd.DataFrame({
    'Variable':        df.columns,
    'Tipo':            [str(df[c].dtype) for c in df.columns],
    'Nulos (%)':       [(df[c].isnull().sum()/len(df)*100).round(2) for c in df.columns],
    'Duplicados':      [f'{n_dup} ({pct_dup:.4f}%)' if i == 0 else '—'
                        for i, c in enumerate(df.columns)],
    'Inconsistencias': [
        str(int((df[c] < 0).sum()))
        if pd.api.types.is_numeric_dtype(df[c]) and c not in ('isFraud', 'isFlaggedFraud')
        else '—'
        for c in df.columns
    ],
    'Acción tomada': [acciones[c] for c in df.columns],
})
print(resumen_cal.to_string(index=False))

print("""
Interpretación del estado de calidad:
El dataset PaySim muestra una calidad muy alta. No existen valores nulos
en ninguna columna, lo que elimina la necesidad de imputación.
No se detectaron filas duplicadas exactas, ni montos o balances negativos.
Las columnas nameOrig, nameDest (IDs sin valor analítico) y step (tiempo
de simulación sin poder discriminante) fueron eliminadas del análisis.
Los balances inconsistentes se conservaron intencionalmente como señales
de posible fraude. El único riesgo real es el fuerte desbalance de clases
en 'isFraud' (~0.14% fraude), que debe considerarse en todo modelo predictivo.
""")
salto()


# ══════════════════════════════════════════════════════════
#  Variables activas para el resto del análisis
# ══════════════════════════════════════════════════════════
numericas = ['amount', 'oldbalanceOrg', 'newbalanceOrig',
             'oldbalanceDest', 'newbalanceDest']


# ── 3.1 — describe() + CV, skew, kurtosis ───────────────
print("3.1 — Estadística descriptiva extendida\n")
desc = df[numericas].describe().round(2)
desc.loc['cv']       = (df[numericas].std() / df[numericas].mean()).round(4)
desc.loc['skewness'] = df[numericas].skew().round(4)
desc.loc['kurtosis'] = df[numericas].kurtosis().round(4)
print(desc.T.to_string())
salto()


# ── 3.2 — Interpretación por variable ───────────────────
print("3.2 — Interpretación variable por variable\n")

# Textos de interpretación usan valores calculados dinámicamente
for col in numericas:
    mean_v  = df[col].mean()
    median_v = df[col].median()
    cv_v    = df[col].std() / mean_v if mean_v != 0 else 0
    skew_v  = df[col].skew()
    kurt_v  = df[col].kurtosis()

    print(f"→ {col.upper()}")
    print(f"   Media: {mean_v:>15,.2f}  |  Mediana: {median_v:>15,.2f}")
    print(f"   CV: {cv_v:.4f}  |  Skew: {skew_v:.4f}  |  Kurtosis: {kurt_v:.4f}")

    if mean_v != 0 and abs(mean_v - median_v) / abs(mean_v) > 0.1:
        print("   ⚠️  Media y mediana difieren significativamente → distribución asimétrica")
    if abs(skew_v) > 1:
        print(f"   ⚠️  Asimetría FUERTE (skew = {skew_v:.2f})")
    if kurt_v > 3:
        print(f"   ⚠️  Colas pesadas (kurtosis = {kurt_v:.2f})")

    # Interpretaciones contextuales dinámicas
    if col == 'amount':
        print(f"   La media (${mean_v:,.0f}) supera ampliamente a la mediana (${median_v:,.0f}): "
              f"la distribución está sesgada a la derecha por transacciones de muy alto valor. "
              f"CV = {cv_v:.2f} → variabilidad extrema. "
              f"Asimetría muy fuerte (skew ≈ {skew_v:.1f}). Colas pesadas (kurtosis >> 3).")
    elif col == 'oldbalanceOrg':
        print(f"   Media (${mean_v:,.0f}) >> mediana (${median_v:,.0f}): la mayoría de cuentas "
              f"tienen saldo bajo pero algunas tienen saldos millonarios. CV = {cv_v:.2f}.")
    elif col == 'newbalanceOrig':
        print(f"   Patrón similar a oldbalanceOrg. Mediana = ${median_v:,.0f}: muchas transacciones "
              f"vacían la cuenta, coherente con fraudes de tipo CASH_OUT.")
    elif col == 'oldbalanceDest':
        print(f"   Mediana = ${median_v:,.0f}. Muchos destinatarios tienen saldo bajo antes "
              f"de recibir (cuentas mula). CV = {cv_v:.2f} → gran dispersión.")
    elif col == 'newbalanceDest':
        print(f"   Similar a oldbalanceDest. Patrón de cuentas que acumulan fondos "
              f"puntualmente. Skew = {skew_v:.1f} → cola derecha muy pesada.")
    print()
salto()


# ── 3.3 — Variables categóricas ─────────────────────────
print("3.3 — Variables categóricas\n")

for col in ['type']:
    print(f"Variable: {col}")
    print(f"  Categorías únicas: {df[col].nunique()}\n")

    fa = df[col].value_counts()
    fr = df[col].value_counts(normalize=True).mul(100).round(4)

    resumen_cat = pd.DataFrame({'Frecuencia absoluta': fa, 'Frecuencia relativa (%)': fr})
    print(resumen_cat.to_string())
    print()

    raras = fr[fr < 1]
    if not raras.empty:
        print("  ⚠️ Categorías con < 1% de representación:")
        print(raras.to_string())
    else:
        print("  No hay categorías con menos del 1% de representación.")

    print("""
  Nota sobre Frecuencia absoluta vs relativa:
    - Frecuencia absoluta: cantidad de filas en esa categoría (conteo directo).
    - Frecuencia relativa: porcentaje que representa esa categoría del total.
    Ambas indican qué tan común es cada tipo de transacción en la muestra.
    DEBIT tiene muy baja representación → categoría menor pero presente.
""")
salto()


# ── 3.4 — Prevalencia de la variable objetivo ────────────
print("3.4 — Prevalencia de isFraud\n")
vc = df['isFraud'].value_counts().sort_index()
pct = df['isFraud'].value_counts(normalize=True).sort_index().mul(100).round(4)

print(pd.DataFrame({'Conteo': vc, 'Porcentaje (%)': pct}).to_string())
print(f"""
¿Está balanceada? NO. Existe un desbalance severo:
  Clase 0 (no fraude): {vc[0]:,} registros ({pct[0]:.2f}%)
  Clase 1 (fraude)   : {vc[1]:,} registros ({pct[1]:.4f}%)

Implicaciones para un modelo futuro:
  Un clasificador que prediga siempre "no fraude" obtendría ~{pct[0]:.1f}% de
  accuracy sin detectar ningún fraude real. Por ello se deben usar
  métricas como Precision, Recall y F1-Score sobre la clase 1, junto con
  técnicas de balanceo como SMOTE, undersampling o ajuste de class_weight.
""")

fig, ax = plt.subplots(figsize=(7, 4))
vc.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='white')
ax.set_title('Distribución de isFraud — clase objetivo', fontsize=12)
ax.set_xlabel('isFraud', fontsize=10)
ax.set_ylabel('Cantidad de transacciones', fontsize=10)
ax.set_xticklabels(['0 — No fraude', '1 — Fraude'], rotation=0)
for i, (idx, v) in enumerate(vc.items()):
    pct_val = v / len(df) * 100
    ax.text(i, v + len(df)*0.002,
            f'{v:,}\n({pct_val:.2f}%)', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('3_4_isfraud_balance.png', dpi=120, bbox_inches='tight')
plt.show()
salto()


# ── 3.4b — Top 3 variables con mayor variabilidad (CV) ──
print("3.4b — Top 3 variables con mayor variabilidad (CV)\n")
cv_series = (df[numericas].std() / df[numericas].mean()).sort_values(ascending=False)
top3 = cv_series.head(3)
print(top3.to_string())

# Interpretación dinámica según ranking real
print("\nInterpretación:")
for rank, (col, cv_val) in enumerate(top3.items(), 1):
    skew_v = df[col].skew()
    if col == 'oldbalanceOrg':
        print(f"  {rank}. {col} (CV = {cv_val:.2f}): Los saldos iniciales del originador varían "
              f"enormemente (skew = {skew_v:.1f}). En fraudes las cuentas suelen estar "
              f"llenas justo antes del vaciado.")
    elif col == 'newbalanceOrig':
        print(f"  {rank}. {col} (CV = {cv_val:.2f}): Muchas transacciones dejan el saldo en 0 "
              f"(vaciado completo de cuenta). Alta dispersión (skew = {skew_v:.1f}).")
    elif col == 'amount':
        print(f"  {rank}. {col} (CV = {cv_val:.2f}): Los montos van de <$1 a millones. "
              f"Alta variabilidad coherente con mezcla de pagos cotidianos y "
              f"transferencias de alto valor propias del fraude (skew = {skew_v:.1f}).")
    elif col == 'oldbalanceDest':
        print(f"  {rank}. {col} (CV = {cv_val:.2f}): Saldo del destinatario antes de recibir. "
              f"Cuentas mula suelen tener saldo bajo o cero (skew = {skew_v:.1f}).")
    elif col == 'newbalanceDest':
        print(f"  {rank}. {col} (CV = {cv_val:.2f}): El saldo post-transacción del destinatario "
              f"es muy disperso; las cuentas mula acumulan montos atípicos de golpe.")
print()

fig, ax = plt.subplots(figsize=(8, 4))
top3.plot(kind='bar', ax=ax, color=['#3498db', '#9b59b6', '#1abc9c'], edgecolor='white')
ax.set_title('Top 3 variables por coeficiente de variación (CV)', fontsize=12)
ax.set_xlabel('Variable', fontsize=10)
ax.set_ylabel('CV  (std / media)', fontsize=10)
ax.set_xticklabels(top3.index, rotation=0)
for i, v in enumerate(top3):
    ax.text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('3_4_top3_cv.png', dpi=120, bbox_inches='tight')
plt.show()
salto()


# ══════════════════════════════════════════════════════════
#  ACTIVIDAD 4 — DISTRIBUCIONES UNIVARIADAS
# ══════════════════════════════════════════════════════════

continuas = ['amount', 'oldbalanceOrg', 'newbalanceOrig',
             'oldbalanceDest', 'newbalanceDest']

# ── 4.1 — Histogramas KDE en Z-Score ────────────────────
print("4.1 — Histogramas con KDE — variables en Z-Score\n")
print(
    "Todas las variables se grafican en Z-Score (media=0, std=1) para permitir\n"
    "una comparación visual directa entre ellas en la misma escala.\n"
    "El eje X se recorta al p99 del Z-Score de cada variable para mostrar la\n"
    "zona de mayor densidad sin que los outliers extremos compriman el gráfico.\n"
    "Esto NO elimina datos; es solo un ajuste visual.\n"
)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, col in enumerate(continuas):
    ax = axes[i]
    zs = zscore(df[col])
    p01_zs = zs.quantile(0.01)
    p99_zs = zs.quantile(0.99)
    mean_zs   = zs.mean()    # siempre 0 por construcción
    median_zs = zs.median()  # puede diferir de 0 si hay asimetría

    sns.histplot(zs, bins=60, kde=True, ax=ax, color='#3498db', edgecolor='white')
    ax.axvline(mean_zs,   color='red',    linestyle='--', linewidth=1.5,
               label=f'Media: {mean_zs:.2f}')
    ax.axvline(median_zs, color='orange', linestyle='-',  linewidth=1.5,
               label=f'Mediana: {median_zs:.2f}')
    ax.set_xlim(p01_zs - 0.1, p99_zs + 0.1)
    ax.set_title(f'Z-Score({col})\n[p01={p01_zs:.2f} | p99={p99_zs:.2f}]', fontsize=10)
    ax.set_xlabel('Desviaciones estándar (σ)', fontsize=9)
    ax.set_ylabel('Frecuencia', fontsize=9)
    ax.legend(fontsize=8)

axes[-1].set_visible(False)

fig.suptitle('Distribuciones univariadas — variables continuas en Z-Score\n'
             '(eje X recortado entre p01 y p99 para mejor visibilidad)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('4_1_histogramas_kde.png', dpi=120, bbox_inches='tight')
plt.show()

print("""
Observaciones:
  - Media = 0 en todas las variables (por definición del Z-Score).
  - Cuando mediana ≠ 0, indica asimetría: la distribución no es simétrica
    alrededor de su centro, y la mayoría de valores está a un lado de la media.
  - Variables con p99 muy alto (ej. oldbalanceOrg > 5σ) confirman la presencia
    de outliers extremos en la cola derecha, coherente con el skew elevado.
  - La cola derecha visible incluso dentro del zoom p01–p99 confirma que los
    outliers son estructurales en datos financieros, no errores de medición.
""")
salto()


# ── 4.2 — Clasificación de distribuciones ───────────────
print("4.2 — Clasificación de distribuciones\n")

clasif = []
for col in continuas:
    skew_v = df[col].skew()
    if abs(skew_v) < 0.5:
        tipo = "Aproximadamente normal"
    elif skew_v >= 2:
        tipo = "Sesgada fuertemente a la derecha"
    elif skew_v >= 0.5:
        tipo = "Sesgada a la derecha"
    elif skew_v <= -2:
        tipo = "Sesgada fuertemente a la izquierda"
    else:
        tipo = "Sesgada a la izquierda"
    clasif.append({'Variable': col, 'Skewness': round(skew_v, 4), 'Clasificación': tipo})

tabla_dist = pd.DataFrame(clasif)
print(tabla_dist.to_string(index=False))
print("""
  - Todas las variables financieras están fuertemente sesgadas a la derecha.
  - La mayoría de valores son bajos/moderados, pero existen outliers de muy
    alto valor que elevan tanto la media como el skew.
  - Esta forma es típica en distribuciones de montos y saldos financieros.
  - Nota: el Z-Score no cambia el skewness (es una transformación lineal);
    solo reescala los valores para que sean comparables entre variables.
""")
salto()


# ── 4.3 — Estandarización Z-Score: efecto en la escala ──
print("4.3 — Estandarización Z-Score: antes y después\n")
print(
    "El Z-Score es una transformación LINEAL (X' = (X - μ) / σ) que lleva\n"
    "cada variable a media=0, std=1. No cambia la forma de la distribución\n"
    "(skewness y kurtosis se conservan), pero unifica la escala de todas las\n"
    "variables, lo cual es esencial para modelos sensibles a la escala.\n"
    "En las gráficas se aprecia que la FORMA es idéntica antes y después;\n"
    "lo que cambia es el eje X: pasa de unidades originales (USD) a\n"
    "desviaciones estándar (σ), permitiendo comparar variables directamente.\n"
)

skewness_vals = df[continuas].skew()
vars_plot = skewness_vals[skewness_vals > 2].index.tolist()

print("Variables con skew > 2 (las más asimétricas, candidatas a estandarizar):")
print(skewness_vals.round(4).to_string())
print(f"\nVariables seleccionadas para graficar: {vars_plot}\n")

for col in vars_plot:
    col_zs = zscore(df[col])

    # Límites de zoom
    p01_raw = df[col].quantile(0.01)
    p99_raw = df[col].quantile(0.99)
    p01_zs  = col_zs.quantile(0.01)
    p99_zs  = col_zs.quantile(0.99)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ANTES — escala original
    sns.histplot(df[col], bins=60, kde=True, ax=axes[0],
                 color='#3498db', edgecolor='white')
    axes[0].axvline(df[col].mean(),   color='red',    linestyle='--', linewidth=1.5,
                    label=f'Media: ${df[col].mean():,.0f}')
    axes[0].axvline(df[col].median(), color='orange', linestyle='-',  linewidth=1.5,
                    label=f'Mediana: ${df[col].median():,.0f}')
    axes[0].set_xlim(max(0, p01_raw - 0.05*p99_raw), p99_raw)
    axes[0].set_title(f'{col} — ANTES del Z-Score\n(escala USD, zoom p01–p99)', fontsize=11)
    axes[0].set_xlabel(f'{col} (USD)', fontsize=10)
    axes[0].set_ylabel('Frecuencia', fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(
        lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}k' if x >= 1e3 else f'{x:,.0f}'
    ))

    # DESPUÉS — Z-Score
    sns.histplot(col_zs, bins=60, kde=True, ax=axes[1],
                 color='#1abc9c', edgecolor='white')
    axes[1].axvline(col_zs.mean(),   color='red',    linestyle='--', linewidth=1.5,
                    label=f'Media: {col_zs.mean():.2f}')
    axes[1].axvline(col_zs.median(), color='orange', linestyle='-',  linewidth=1.5,
                    label=f'Mediana: {col_zs.median():.2f}')
    axes[1].set_xlim(p01_zs - 0.1, p99_zs + 0.1)
    axes[1].set_title(f'Z-Score({col}) — DESPUÉS\n(escala σ, zoom p01–p99)', fontsize=11)
    axes[1].set_xlabel('Desviaciones estándar (σ)', fontsize=10)
    axes[1].set_ylabel('Frecuencia', fontsize=10)
    axes[1].legend(fontsize=9)

    skew_v = df[col].skew()
    fig.suptitle(
        f'Estandarización Z-Score — {col}  |  skew = {skew_v:.2f} (conservado)\n'
        f'La FORMA es idéntica; cambia únicamente el eje X',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(f'4_3_zscore_{col}.png', dpi=120, bbox_inches='tight')
    plt.show()

    print(f"  {col}: skew = {skew_v:.4f}  |  "
          f"Rango original: [0, {df[col].max():,.0f}]  →  "
          f"Rango Z-Score: [{col_zs.min():.2f}, {col_zs.max():.2f}]")
print()
salto()


# ── 4.4 — Variables categóricas ordenadas ───────────────
print("4.4 — Barras de variables categóricas (mayor a menor)\n")

frecuencias_type = df['type'].value_counts().sort_values(ascending=False)
pct_dom = frecuencias_type.iloc[0] / frecuencias_type.sum() * 100

fig, ax = plt.subplots(figsize=(9, 5))
colores = ['#e74c3c' if i == 0 else '#3498db' for i in range(len(frecuencias_type))]
ax.bar(frecuencias_type.index, frecuencias_type.values,
       color=colores, edgecolor='white')
ax.set_title('Frecuencia de tipos de transacción (mayor a menor)', fontsize=12)
ax.set_xlabel('Tipo de transacción', fontsize=10)
ax.set_ylabel('Cantidad de transacciones', fontsize=10)
for i, v in enumerate(frecuencias_type.values):
    pct_v = v / len(df) * 100
    ax.text(i, v + len(df)*0.002,
            f'{v:,}\n({pct_v:.1f}%)', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('4_4_barras_tipo.png', dpi=120, bbox_inches='tight')
plt.show()

cat_dom = frecuencias_type.index[0]
print(f"Categoría dominante: {cat_dom} ({pct_dom:.1f}%)")
if pct_dom > 35:
    print(f"⚠️  {cat_dom} domina el dataset. Un modelo podría sesgar sus predicciones")
    print(f"   hacia el comportamiento de este tipo de transacción.")

cat_rara = frecuencias_type[frecuencias_type / frecuencias_type.sum() * 100 < 1]
if not cat_rara.empty:
    for cat, cnt in cat_rara.items():
        pct_r = cnt / len(df) * 100
        print(f"\n  ⚠️  {cat} representa solo el {pct_r:.2f}% → categoría minoritaria.")
        print( "     Si se usa como feature en un modelo, podría necesitar")
        print( "     agrupación con otra categoría o tratamiento especial.")
salto()


# ------------------------------------------------------------------

print("4.5 — Distribución de amount separada por isFraud\n")
print(
    "Nota: se usa log1p(amount) en lugar de Z-Score para esta comparación.\n"
    "Razón: al graficar dos grupos (fraude vs legítima), lo relevante es ver\n"
    "la DIFERENCIA de ubicación entre ambas distribuciones. Z-Score aplicado\n"
    "por grupo eliminaría esa diferencia; log1p comprime la escala sin\n"
    "perder la separación entre clases.\n"
)

df['amount_log'] = np.log1p(df['amount'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(
    data=df, x='amount_log', hue='isFraud',
    bins=60, kde=True, stat='density', common_norm=False,
    palette={0: '#2ecc71', 1: '#e74c3c'}, alpha=0.55,
    edgecolor='white', ax=axes[0]
)
axes[0].set_title('Densidad normalizada por clase', fontsize=11)
axes[0].set_xlabel('log1p(amount)', fontsize=10)
axes[0].set_ylabel('Densidad', fontsize=10)
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, ['0 — No fraude', '1 — Fraude'], title='isFraud')

df['isFraud_lbl'] = df['isFraud'].map({0: 'No fraude', 1: 'Fraude'})
orden_lbl = ['No fraude', 'Fraude']
sns.boxplot(
    data=df, x='isFraud_lbl', y='amount_log',
    order=orden_lbl,
    hue='isFraud_lbl', hue_order=orden_lbl,
    palette={'No fraude': '#2ecc71', 'Fraude': '#e74c3c'},
    legend=False, ax=axes[1], width=0.4
)
axes[1].set_title('Boxplot de log1p(amount) por clase', fontsize=11)
axes[1].set_xlabel('Clase', fontsize=10)
axes[1].set_ylabel('log1p(amount)', fontsize=10)

# Anotar medianas reales (en USD) sobre el boxplot
for i, lbl in enumerate(orden_lbl):
    is_fraud_val = 1 if lbl == 'Fraude' else 0
    med_usd = df[df['isFraud'] == is_fraud_val]['amount'].median()
    med_log = np.log1p(med_usd)
    axes[1].text(i, med_log + 0.15,
                 f'Mediana USD:\n${med_usd:,.0f}',
                 ha='center', fontsize=8, color='black',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

plt.suptitle('Monto de transacción (log1p) vs. Clase de fraude', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('4_5_amount_vs_fraud.png', dpi=120, bbox_inches='tight')
plt.show()

# Mostrar estadísticas reales en consola
med_fraud    = df[df['isFraud']==1]['amount'].median()
med_nofraud  = df[df['isFraud']==0]['amount'].median()
mean_fraud   = df[df['isFraud']==1]['amount'].mean()
mean_nofraud = df[df['isFraud']==0]['amount'].mean()
print(f"  amount — Mediana  |  No fraude: ${med_nofraud:>12,.0f}  |  Fraude: ${med_fraud:>12,.0f}")
print(f"  amount — Media    |  No fraude: ${mean_nofraud:>12,.0f}  |  Fraude: ${mean_fraud:>12,.0f}")
print(f"  Ratio de medianas (fraude / no fraude): {med_fraud/med_nofraud:.1f}x\n")
print("""
Interpretación:
  - Las transacciones fraudulentas se concentran en montos significativamente
    más altos que las legítimas, como confirman mediana y media calculadas.
  - El histograma muestra que la distribución de fraude (rojo) tiene su pico
    desplazado hacia valores más altos de log1p(amount).
  - El boxplot hace explícita la diferencia de medianas en escala logarítmica;
    las medianas en USD se anotan directamente sobre la gráfica.
  - 'amount' es una feature discriminante de alto valor para detectar fraude.
""")

df.drop(columns=['amount_log', 'isFraud_lbl'], inplace=True)
salto()


print("Extra — Relación entre tipo de transacción y variables monetarias\n")

vars_dinero = ['amount', 'oldbalanceOrg', 'newbalanceOrig',
               'oldbalanceDest', 'newbalanceDest']

resumen_type = df.groupby('type', observed=True)[vars_dinero].agg(['mean', 'median']).round(0)
print("Estadísticas por tipo de transacción (valores en USD):")
print(resumen_type.to_string())
print()


pivot_mean = df.groupby('type', observed=True)[vars_dinero].mean()
pivot_norm = (pivot_mean - pivot_mean.mean()) / pivot_mean.std()

fig, ax = plt.subplots(figsize=(11, 4))
annot_df = pivot_mean.map(
    lambda x: f'${x/1e6:.1f}M' if x >= 1e6
    else f'${x/1e3:.0f}k' if x >= 1e3
    else f'${x:.0f}'
)
sns.heatmap(
    pivot_norm,
    annot=annot_df, fmt='',
    cmap='RdYlGn', linewidths=0.5, linecolor='white',
    ax=ax, cbar_kws={'label': 'Z-Score de la media (relativo a otras categorías)'}
)
ax.set_title('Media de variables monetarias por tipo de transacción\n'
             'Color = Z-Score (posición relativa entre tipos)  |  Anotación = valor real en USD',
             fontsize=11)
ax.set_xlabel('Variable monetaria', fontsize=10)
ax.set_ylabel('Tipo de transacción', fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig('extra_heatmap_type_dinero.png', dpi=120, bbox_inches='tight')
plt.show()

df['amount_log'] = np.log1p(df['amount'])

fig, ax = plt.subplots(figsize=(10, 5))
orden_tipos = (df.groupby('type', observed=True)['amount_log']
               .median().sort_values(ascending=False).index)
sns.boxplot(
    data=df, x='type', y='amount_log', hue='type',
    order=orden_tipos, palette='Set2', legend=False, ax=ax
)
# Anotar mediana en USD sobre cada boxplot
for i, tipo in enumerate(orden_tipos):
    med_usd = df[df['type'] == tipo]['amount'].median()
    med_log = np.log1p(med_usd)
    ax.text(i, med_log + 0.1, f'${med_usd/1e3:.0f}k',
            ha='center', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.75))

ax.set_title('Distribución de log1p(amount) por tipo de transacción\n'
             '(ordenado por mediana descendente; anotación = mediana en USD)',
             fontsize=12)
ax.set_xlabel('Tipo de transacción', fontsize=10)
ax.set_ylabel('log1p(amount)', fontsize=10)
plt.tight_layout()
plt.savefig('extra_boxplot_type_amount.png', dpi=120, bbox_inches='tight')
plt.show()

df.drop(columns=['amount_log'], inplace=True)

print("""
Interpretación:
  - El heatmap confirma que TRANSFER maneja los montos (amount) más altos
    en términos relativos (verde intenso), seguido de CASH_OUT. Son los tipos
    de transacción donde se concentra el fraude.
  - CASH_IN destaca por los saldos del destinatario (oldbalanceDest,
    newbalanceDest) más elevados: los fondos llegan a cuentas ya activas.
  - PAYMENT opera montos muy bajos y no tiene saldo de destinatario registrado
    (el destinatario es un comercio, no una cuenta bancaria individual).
  - El boxplot refuerza la separación: TRANSFER tiene la mediana de monto más
    alta (~$497k), mientras DEBIT y PAYMENT se ubican muy por debajo (~$3k–$9k).
  - Esta diferencia de escala entre tipos de transacción hace que 'type' sea
    una de las features más discriminantes del dataset.
""")
salto()



print("EXTRA — Análisis de normalización: Min-Max vs Z-Score\n")

print("""
¿Para qué sirve normalizar?
  La normalización no cambia la forma de la distribución ni mejora el EDA,
  pero es NECESARIA para modelos sensibles a la escala de los datos:
  - Regresión logística, SVM, K-NN, redes neuronales, KMeans → requieren escala.
  - Árboles de decisión, Random Forest, XGBoost → NO necesitan escala.

Min-Max Scaling  →  X' = (X - Xmin) / (Xmax - Xmin)
  - Resultado: rango [0, 1].
  - Problema: un solo outlier extremo determina Xmax y comprime el 99%
    de los datos cerca de 0. NO recomendado cuando hay outliers fuertes.

Z-Score (Standardización)  →  X' = (X - μ) / σ
  - Resultado: media = 0, std = 1 (sin rango fijo).
  - Más robusto frente a outliers que Min-Max.
  - No elimina outliers, pero los expresa en unidades de desviación estándar.

Recomendación por variable:
""")

print(f"{'Variable':<18} {'Método recomendado':<22} {'Justificación'}")
print("-" * 100)
recomendaciones = {
    'amount':         'Skew = {:.1f}. Outliers extremos; Min-Max colapsaría la escala.',
    'oldbalanceOrg':  'Skew = {:.1f}. Outliers fuertes. Z-Score más robusto.',
    'newbalanceOrig': 'Skew = {:.1f}. Muchos valores en 0. Z-Score los maneja correctamente.',
    'oldbalanceDest': 'Skew = {:.1f}. Cuentas con saldo 0. Misma justificación.',
    'newbalanceDest': 'Skew = {:.1f}. Igual que oldbalanceDest.',
}
for col, tmpl in recomendaciones.items():
    justif = tmpl.format(df[col].skew())
    print(f"  {col:<16} {'Z-Score':<22} {justif}")

print("""
Columnas excluidas de normalización:
  isFraud, isFlaggedFraud → binarias (0/1); no se normalizan.
  type                    → categórica; se codifica (OneHotEncoding),
                            no se normaliza.
  nameOrig, nameDest, step → ya eliminados del dataset.
""")

# Demo visual: comparación de las 3 escalas para 'amount'
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Original (con zoom p99)
p99_raw = df['amount'].quantile(0.99)
sns.histplot(df['amount'], bins=50, kde=True, ax=axes[0], color='#3498db', edgecolor='white')
axes[0].set_xlim(0, p99_raw)
axes[0].set_title(f'amount  (sin normalizar)\nzoom p99 = ${p99_raw:,.0f}', fontsize=10)
axes[0].set_xlabel('USD', fontsize=9); axes[0].set_ylabel('Frecuencia', fontsize=9)
axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(
    lambda x, _: f'{x/1e3:.0f}k' if x >= 1e3 else f'{x:.0f}'
))

# Min-Max (con zoom p99 del escalado)
mm = (df['amount'] - df['amount'].min()) / (df['amount'].max() - df['amount'].min())
p99_mm = mm.quantile(0.99)
sns.histplot(mm, bins=50, kde=True, ax=axes[1], color='#e67e22', edgecolor='white')
axes[1].set_xlim(0, p99_mm)
axes[1].set_title(f'Min-Max(amount)\nzoom p99 = {p99_mm:.5f}', fontsize=10)
axes[1].set_xlabel('Valor escalado [0–1]', fontsize=9); axes[1].set_ylabel('Frecuencia', fontsize=9)
axes[1].annotate(
    f'El 99% de los datos\ncae entre 0 y {p99_mm:.4f}',
    xy=(p99_mm*0.5, axes[1].get_ylim()[1]*0.5),
    fontsize=8, color='#e67e22', ha='center',
    bbox=dict(boxstyle='round', fc='white', alpha=0.8)
)

# Z-Score (zoom p01–p99)
zs = zscore(df['amount'])
p01_zs = zs.quantile(0.01); p99_zs = zs.quantile(0.99)
sns.histplot(zs, bins=50, kde=True, ax=axes[2], color='#1abc9c', edgecolor='white')
axes[2].set_xlim(p01_zs - 0.1, p99_zs + 0.1)
axes[2].axvline(0, color='red', linestyle='--', linewidth=1.2, label='Media = 0')
axes[2].legend(fontsize=8)
axes[2].set_title(f'Z-Score(amount)\nzoom p01={p01_zs:.2f} / p99={p99_zs:.2f}', fontsize=10)
axes[2].set_xlabel('Desviaciones estándar (σ)', fontsize=9); axes[2].set_ylabel('Frecuencia', fontsize=9)

plt.suptitle('Comparación de métodos de normalización — amount\n'
             '(Min-Max colapsa el 99% de los datos; Z-Score distribuye mejor la escala)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('extra_normalizacion_amount.png', dpi=120, bbox_inches='tight')
plt.show()

print("Gráfico de normalización generado.")
salto()


# ── Categorización de amount: Monto Bajo / Monto Alto ──────────────
# Corte en la mediana (P50 = ~$76,031): estadísticamente robusto frente
# al skew = 22 de amount, y genera grupos perfectamente balanceados (50/50).

corte = df['amount'].median()  # 76,030.86

df['amount_categoria'] = pd.cut(
    df['amount'],
    bins=[0, corte, float('inf')],
    labels=['Monto Bajo', 'Monto Alto'],
    include_lowest=True       # incluye el valor mínimo (0.92) en el primer bin
)

print(f"Corte aplicado (mediana): ${corte:,.2f}\n")
print(df['amount_categoria'].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colores = {'Monto Bajo': '#3498db', 'Monto Alto': '#e74c3c'}

# ── Subplot 1: histograma con KDE y línea de corte ─────────────────
amount_log = np.log1p(df['amount'])

for categoria, grupo in df.groupby('amount_categoria', observed=True):
    sns.kdeplot(
        np.log1p(grupo['amount']),
        ax=axes[0],
        label=categoria,
        color=colores[categoria],
        fill=True,
        alpha=0.35,
        linewidth=2
    )

axes[0].axvline(
    np.log1p(corte),
    color='black',
    linestyle='--',
    linewidth=1.8,
    label=f'Corte: ${corte:,.0f}'
)
axes[0].set_title('Distribución de amount por categoría\n(escala log1p)', fontsize=11)
axes[0].set_xlabel('log1p(amount)', fontsize=10)
axes[0].set_ylabel('Densidad', fontsize=10)
axes[0].legend(fontsize=9)

# ── Subplot 2: tasa de fraude por categoría ────────────────────────
fraude_pct = (
    df.groupby('amount_categoria', observed=True)['isFraud']
    .mean()
    .mul(100)
    .reset_index()
)

bars = axes[1].bar(
    fraude_pct['amount_categoria'],
    fraude_pct['isFraud'],
    color=[colores[c] for c in fraude_pct['amount_categoria']],
    edgecolor='white',
    width=0.5
)

for bar, val in zip(bars, fraude_pct['isFraud']):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.005,
        f'{val:.3f}%',
        ha='center',
        fontsize=10,
        fontweight='bold'
    )

axes[1].set_title('Tasa de fraude por categoría de monto', fontsize=11)
axes[1].set_xlabel('Categoría', fontsize=10)
axes[1].set_ylabel('% Transacciones fraudulentas', fontsize=10)
axes[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f%%'))

plt.suptitle(
    f'Segmentación de amount — corte en mediana (${corte:,.0f})',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('amount_categorias.png', dpi=120, bbox_inches='tight')
plt.show()
