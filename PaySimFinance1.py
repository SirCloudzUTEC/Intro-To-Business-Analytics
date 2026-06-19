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

print("Extra — Tasa de fraude por tipo de transacción\n")

fraude_tipo = (
    df.groupby('type', observed=True)
    .agg(
        total=('isFraud', 'count'),
        fraudes=('isFraud', 'sum'),
        tasa_fraude=('isFraud', 'mean')
    )
    .reset_index()
)

fraude_tipo['tasa_fraude'] *= 100
fraude_tipo = fraude_tipo.sort_values('tasa_fraude', ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(
    data=fraude_tipo,
    x='type',
    y='tasa_fraude',
    ax=ax
)

ax.set_title('Tasa de fraude por tipo de transacción', fontsize=12)
ax.set_xlabel('Tipo de transacción')
ax.set_ylabel('% de fraude')

for i, row in enumerate(fraude_tipo.itertuples()):
    ax.text(
        i,
        row.tasa_fraude + 0.01,
        f'{row.tasa_fraude:.3f}%',
        ha='center',
        fontweight='bold'
    )

plt.tight_layout()
plt.show()

print("""
Interpretación:
TRANSFER y CASH_OUT deben revisarse con mayor atención porque concentran
los patrones más asociados al fraude. En cambio, PAYMENT, CASH_IN y DEBIT
presentan menor riesgo relativo dentro de la muestra.
""")
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

# ── Categorización de amount: Monto Bajo / Monto Alto ──────────────

print("EXTRA — Rangos de monto: Bajo vs Alto\n")

# Corte usando la mediana
corte = df['amount'].median()

df['amount_categoria'] = pd.cut(
    df['amount'],
    bins=[0, corte, float('inf')],
    labels=['Monto Bajo', 'Monto Alto'],
    include_lowest=True
)

print(f"Corte aplicado usando la mediana: ${corte:,.2f}\n")

print("Cantidad de transacciones por categoría:")
print(df['amount_categoria'].value_counts())
print()

# Tabla descriptiva de rangos
rango_montos = (
    df.groupby('amount_categoria', observed=True)
    .agg(
        monto_minimo=('amount', 'min'),
        monto_maximo=('amount', 'max'),
        monto_mediano=('amount', 'median'),
        cantidad=('amount', 'count'),
        fraudes=('isFraud', 'sum'),
        tasa_fraude=('isFraud', 'mean')
    )
    .reset_index()
)

rango_montos['tasa_fraude'] = rango_montos['tasa_fraude'] * 100

print("Descripción de rangos de monto:")
print(rango_montos.to_string(index=False))
print()

colores = {
    'Monto Bajo': '#3498db',
    'Monto Alto': '#e74c3c'
}

# ── Gráfico 1: KDE detallado por categoría ─────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))

for categoria, grupo in df.groupby('amount_categoria', observed=True):

    monto_min = grupo['amount'].min()
    monto_max = grupo['amount'].max()
    monto_med = grupo['amount'].median()

    sns.kdeplot(
        np.log1p(grupo['amount']),
        ax=ax,
        label=(
            f"{categoria}\n"
            f"Min: ${monto_min:,.0f}\n"
            f"Mediana: ${monto_med:,.0f}\n"
            f"Max: ${monto_max:,.0f}"
        ),
        color=colores[categoria],
        fill=True,
        alpha=0.35,
        linewidth=2
    )

ax.axvline(
    np.log1p(corte),
    color='black',
    linestyle='--',
    linewidth=1.8,
    label=f'Corte mediana: ${corte:,.0f}'
)

ax.set_title('Distribución KDE de amount por categoría')
ax.set_xlabel('log1p(amount)')
ax.set_ylabel('Densidad')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('kde_amount_categoria.png', dpi=120, bbox_inches='tight')
plt.show()


# ── Gráfico 2: Tasa de fraude por categoría ────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    rango_montos['amount_categoria'],
    rango_montos['tasa_fraude'],
    color=[colores[c] for c in rango_montos['amount_categoria']],
    edgecolor='white',
    width=0.5
)

for bar, val in zip(bars, rango_montos['tasa_fraude']):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.005,
        f'{val:.3f}%',
        ha='center',
        fontsize=10,
        fontweight='bold'
    )

ax.set_title('Tasa de fraude por rango de monto')
ax.set_xlabel('Categoría de monto')
ax.set_ylabel('% Transacciones fraudulentas')
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f%%'))

plt.tight_layout()
plt.savefig('fraude_por_categoria.png', dpi=120, bbox_inches='tight')
plt.show()


# ── Interpretación final ───────────────────────────────────────────

bajo = rango_montos[rango_montos['amount_categoria'] == 'Monto Bajo'].iloc[0]
alto = rango_montos[rango_montos['amount_categoria'] == 'Monto Alto'].iloc[0]

print(f"""
Interpretación:

- Monto Bajo:
  Va desde ${bajo['monto_minimo']:,.2f} hasta ${bajo['monto_maximo']:,.2f}.
  Contiene {int(bajo['cantidad']):,} transacciones.
  Fraudes detectados: {int(bajo['fraudes'])}.
  Tasa de fraude: {bajo['tasa_fraude']:.3f}%.

- Monto Alto:
  Va desde ${alto['monto_minimo']:,.2f} hasta ${alto['monto_maximo']:,.2f}.
  Contiene {int(alto['cantidad']):,} transacciones.
  Fraudes detectados: {int(alto['fraudes'])}.
  Tasa de fraude: {alto['tasa_fraude']:.3f}%.

Conclusión:
La división por rangos permite ordenar el análisis y comparar si el fraude
se concentra más en transacciones pequeñas o grandes. El primer gráfico KDE
muestra la distribución de montos en escala logarítmica, mientras que el
segundo gráfico compara directamente la tasa de fraude por categoría.
""")

salto()
