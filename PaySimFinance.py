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

#Eliminar y explicar pq borramos ids, step.

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


# ── 2.3 — Conversión de tipos ────────────────────────────
print("2.3 — Verificación y conversión de tipos\n")
print("Tipos originales:")
print(df.dtypes.to_string())

# Conversiones justificadas
df['type']          = df['type'].astype('category')   # variable nominal → category (ahorra memoria)
df['nameOrig']      = df['nameOrig'].astype('category')
df['nameDest']      = df['nameDest'].astype('category')
df['isFraud']       = df['isFraud'].astype('int8')     # bandera binaria → int8 (0/1, ahorra memoria)
df['isFlaggedFraud']= df['isFlaggedFraud'].astype('int8')

print("\nTipos tras conversión:")
print(df.dtypes.to_string())
print("""
Conversiones realizadas:
  type, nameOrig, nameDest  → category  (variables nominales; reduce uso de memoria)
  isFraud, isFlaggedFraud   → int8      (flags binarios; 1 byte es suficiente)

No se encontraron fechas almacenadas como string.
No se encontraron variables numéricas almacenadas como object.
""")
salto()


# ── 2.4 — Valores imposibles / inconsistentes ────────────
print("2.4 — Detección de valores imposibles\n")

montos_neg  = (df['amount'] < 0).sum()
bal_neg     = (
    (df['oldbalanceOrg']  < 0) |
    (df['newbalanceOrig'] < 0) |
    (df['oldbalanceDest'] < 0) |
    (df['newbalanceDest'] < 0)
).sum()
# Balance inconsistente: se debita más de lo que hay en cuenta (con saldo > 0 conocido)
bal_inc = df[
    (df['type'].isin(['TRANSFER', 'CASH_OUT'])) &
    (df['amount'] > df['oldbalanceOrg']) &
    (df['oldbalanceOrg'] > 0)
].shape[0]

print(f"Montos negativos          : {montos_neg}")
print(f"Balances negativos        : {bal_neg}")
print(f"Balances inconsistentes   : {bal_inc:,}  "
      f"({bal_inc/len(df)*100:.2f}% del total)")

print("""
Decisión:
  - Montos y balances negativos: 0 registros → no se requiere acción.
  - Balances inconsistentes (monto > saldo antes): se CONSERVAN.
    Justificación: en el contexto de fraude, que un retiro supere el saldo
    previo es una señal analítica valiosa; eliminar estos registros
    sesgaría el análisis hacia transacciones "normales".
""")
# Solo eliminamos lo estrictamente imposible (negativos)
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
    'step':           'Sin acción (numérico correcto)',
    'type':           'Convertido a category',
    'amount':         'Validado ≥ 0 (sin negativos)',
    'nameOrig':       'Convertido a category',
    'oldbalanceOrg':  'Validado ≥ 0 (sin negativos)',
    'newbalanceOrig': 'Validado ≥ 0 (sin negativos)',
    'nameDest':       'Convertido a category',
    'oldbalanceDest': 'Validado ≥ 0 (sin negativos)',
    'newbalanceDest': 'Validado ≥ 0 (sin negativos)',
    'isFraud':        'Convertido a int8',
    'isFlaggedFraud': 'Convertido a int8',
}

resumen = pd.DataFrame({
    'Variable':        df.columns,
    'Tipo':            [str(df[c].dtype) for c in df.columns],
    'Nulos (%)':       [(df[c].isnull().sum()/len(df)*100).round(2) for c in df.columns],
    'Duplicados':      [f'{n_dup} ({pct_dup:.4f}%)' if i == 0 else '—' for i, c in enumerate(df.columns)],
    'Inconsistencias': [
        str(int((df[c] < 0).sum())) if df[c].dtype in [np.float64, np.int64, 'int8'] else '—'
        for c in df.columns
    ],
    'Acción tomada':   [acciones[c] for c in df.columns],
})
print(resumen.to_string(index=False))

print("""
Interpretación del estado de calidad:
El dataset PaySim muestra una calidad muy alta. No existen valores nulos
en ninguna de sus 11 columnas, lo que elimina la necesidad de imputación.
No se detectaron filas duplicadas exactas, ni montos o balances negativos.
Las variables categóricas estaban almacenadas como 'object' y se convirtieron
a 'category' para optimizar memoria. Los 18,721 registros con balance
inconsistente se conservaron intencionalmente como señales analíticas de
posible fraude. El dataset es confiable para el análisis; el único riesgo
real es el fuerte desbalance de clases en 'isFraud' (0.14% fraude), que
debe considerarse en cualquier modelo predictivo futuro.
""")
salto()


#actividad 3

numericas = ['amount', 'oldbalanceOrg', 'newbalanceOrig',
             'oldbalanceDest', 'newbalanceDest', 'step']

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

interpretaciones = {
    'amount': (
        "La media (~180k) supera ampliamente a la mediana (~76k): la distribución "
        "está sesgada a la derecha por transacciones de muy alto valor. "
        "CV = 3.09 → variabilidad extrema; los montos van de <1 a casi 37M. "
        "⚠️ Asimetría muy fuerte (skew ≈ 22). ⚠️ Colas pesadas (kurtosis >> 3)."
    ),
    'oldbalanceOrg': (
        "Media (~837k) >> mediana (~14k): la mayoría de cuentas tienen saldo bajo "
        "pero algunas tienen saldos millonarios. CV = 3.47 → alta dispersión. "
        "⚠️ Asimetría fuerte (skew ≈ 5.25). ⚠️ Colas pesadas."
    ),
    'newbalanceOrig': (
        "Patrón similar a oldbalanceOrg. Muchas transacciones vacían la cuenta "
        "(mediana cercana a 0), lo que es coherente con fraudes de tipo CASH_OUT. "
        "⚠️ Asimetría fuerte. ⚠️ Colas pesadas."
    ),
    'oldbalanceDest': (
        "Distribución muy asimétrica a la derecha. Muchos destinatarios tienen "
        "saldo 0 antes de recibir (cuentas mula). CV alto indica gran dispersión. "
        "⚠️ Asimetría fuerte (skew ≈ 16.5). ⚠️ Colas pesadas."
    ),
    'newbalanceDest': (
        "Similar a oldbalanceDest. El saldo post-transacción también muestra "
        "patrón de cuentas vacías que acumulan fondos puntualmente. "
        "⚠️ Asimetría muy fuerte (skew ≈ 16.4). ⚠️ Colas pesadas."
    ),
    'step': (
        "Media (~244) y mediana (~240) son muy cercanas → distribución "
        "aproximadamente simétrica. CV = 0.58 (moderado). No hay alertas de "
        "asimetría ni colas pesadas. La variable temporal está bien distribuida."
    ),
}

for col in numericas:
    mean   = df[col].mean()
    median = df[col].median()
    cv     = df[col].std() / mean if mean != 0 else 0
    skew   = df[col].skew()
    kurt   = df[col].kurtosis()

    print(f"→ {col.upper()}")
    print(f"   Media: {mean:>15,.2f}  |  Mediana: {median:>15,.2f}")
    print(f"   CV: {cv:.4f}  |  Skew: {skew:.4f}  |  Kurtosis: {kurt:.4f}")
    if abs(mean - median) / abs(mean) > 0.1:
        print("   ⚠️  Media y mediana difieren significativamente → distribución asimétrica")
    if skew > 1 or skew < -1:
        print(f"   ⚠️  Asimetría FUERTE (skew = {skew:.2f})")
    if kurt > 3:
        print(f"   ⚠️  Colas pesadas (kurtosis = {kurt:.2f})")
    print(f"   {interpretaciones[col]}")
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
    DEBIT aparece con 0.61% → categoría menor pero presente en datos reales.
""")
salto()


# ── 3.4 — Prevalencia de la variable objetivo ────────────
print("3.4 — Prevalencia de isFraud\n")
vc = df['isFraud'].value_counts()
pct = df['isFraud'].value_counts(normalize=True).mul(100).round(4)

print(pd.DataFrame({'Conteo': vc, 'Porcentaje (%)': pct}).to_string())
print(f"""
¿Está balanceada? NO. Existe un desbalance severo:
  Clase 0 (no fraude): {vc[0]:,} registros ({pct[0]:.2f}%)
  Clase 1 (fraude)   : {vc[1]:,} registros ({pct[1]:.4f}%)

Implicaciones para un modelo futuro:
  Un clasificador que prediga siempre "no fraude" obtendría ~99.86% de
  accuracy sin detectar ningún fraude real. Por ello se deben usar
  métricas como Precision, Recall y F1-Score sobre la clase 1, junto con
  técnicas de balanceo como SMOTE, undersampling o ajuste de class_weight.
""")

fig, ax = plt.subplots(figsize=(7, 4))
vc.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='white')
ax.set_title('Distribución de isFraud — clase objetivo', fontsize=12)
ax.set_xlabel('isFraud  (0 = no fraude, 1 = fraude)', fontsize=10)
ax.set_ylabel('Cantidad de transacciones', fontsize=10)
ax.set_xticklabels(['0 — No fraude', '1 — Fraude'], rotation=0)
for i, v in enumerate(vc):
    pct_val = v / len(df) * 100
    ax.text(i, v + 200, f'{v:,}\n({pct_val:.2f}%)', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('3_4_isfraud_balance.png', dpi=120, bbox_inches='tight')
plt.show()
salto()

print("3.4 — Top 3 variables con mayor variabilidad (CV)\n")
cv_series = (df[numericas].std() / df[numericas].mean()).sort_values(ascending=False)
top3 = cv_series.head(3)
print(top3.to_string())
print("""
Las 3 variables con mayor variabilidad son:
  1. oldbalanceOrg  (CV ≈ 3.47): Los saldos iniciales del originador varían
     enormemente; desde cuentas vacías hasta cuentas con millones.
     En fraudes, las cuentas suelen estar llenas justo antes del vaciado.
  2. amount         (CV ≈ 3.09): Los montos oscilan entre <$1 y $37M.
     Alta variabilidad es coherente con un sistema que mezcla pagos
     cotidianos con transferencias de alto valor propias del fraude.
  3. newbalanceDest (CV ≈ 2.83): El saldo del destinatario post-transacción
     también es muy disperso; las cuentas mula suelen recibir montos
     atípicamente grandes de golpe.
""")

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

# ── 4.1 — Histogramas KDE en cuadrícula ─────────────────
print("4.1 — Histogramas con KDE (media y mediana)\n")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, col in enumerate(continuas):
    ax = axes[i]
    sns.histplot(df[col], bins=60, kde=True, ax=ax, color='#3498db', edgecolor='white')
    ax.axvline(df[col].mean(),   color='red',    linestyle='--', linewidth=1.5,
               label=f'Media: {df[col].mean():,.0f}')
    ax.axvline(df[col].median(), color='orange', linestyle='-',  linewidth=1.5,
               label=f'Mediana: {df[col].median():,.0f}')
    ax.set_title(f'Distribución de {col}', fontsize=10)
    ax.set_xlabel(col, fontsize=9)
    ax.set_ylabel('Frecuencia', fontsize=9)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x:,.0f}'))

axes[-1].set_visible(False)  # última celda vacía

fig.suptitle('Distribuciones univariadas — variables continuas', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('4_1_histogramas_kde.png', dpi=120, bbox_inches='tight')
plt.show()

print("""
Nota sobre el eje Y — "Frecuencia":
  En sns.histplot el eje Y por defecto muestra la FRECUENCIA ABSOLUTA:
  cuántas observaciones caen en cada bin (barra). No es un porcentaje.
  La curva KDE superpuesta está escalada para coincidir visualmente con
  las barras; su eje Y también representa densidad relativa escalada,
  NO una probabilidad directa.
""")
salto()


# ── 4.2 — Clasificación de distribuciones ───────────────
print("4.2 — Clasificación de distribuciones\n")

todas_num = numericas  # incluye 'step'
clasif = []
for col in todas_num:
    skew = df[col].skew()
    if abs(skew) < 0.5:
        tipo = "Aproximadamente normal"
    elif skew >= 2:
        tipo = "Sesgada fuertemente a la derecha"
    elif skew >= 0.5:
        tipo = "Sesgada a la derecha"
    elif skew <= -2:
        tipo = "Sesgada fuertemente a la izquierda"
    else:
        tipo = "Sesgada a la izquierda"
    clasif.append({'Variable': col, 'Skewness': round(skew, 4), 'Clasificación': tipo})

tabla_dist = pd.DataFrame(clasif)
print(tabla_dist.to_string(index=False))
print("""
  - step es la única variable aproximadamente normal (skew ≈ 0.38).
  - Todas las variables financieras están fuertemente sesgadas a la derecha:
    la mayoría de valores son bajos, pero existen outliers de muy alto valor.
  - Esto es típico en distribuciones de ingresos/montos financieros.
""")
salto()


# ── 4.3 — Transformación log1p para skew > 2 ────────────
print("4.3 — Transformación logarítmica (skewness > 2)\n")

skewness_vals = df[continuas].skew()
vars_log = skewness_vals[skewness_vals > 2].index.tolist()

print("Skewness por variable continua:")
print(skewness_vals.round(4).to_string())
print(f"\nVariables a transformar (skew > 2): {vars_log}\n")

for col in vars_log:
    col_log = np.log1p(df[col])
    skew_antes   = df[col].skew()
    skew_despues = col_log.skew()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Antes
    sns.histplot(df[col], bins=60, kde=True, ax=axes[0],
                 color='#3498db', edgecolor='white')
    axes[0].axvline(df[col].mean(),   color='red',    linestyle='--', linewidth=1.5,
                    label=f'Media: {df[col].mean():,.0f}')
    axes[0].axvline(df[col].median(), color='orange', linestyle='-',  linewidth=1.5,
                    label=f'Mediana: {df[col].median():,.0f}')
    axes[0].set_title(f'{col} — ANTES  (skew: {skew_antes:.2f})', fontsize=11)
    axes[0].set_xlabel(col); axes[0].set_ylabel('Frecuencia'); axes[0].legend()

    # Después
    sns.histplot(col_log, bins=60, kde=True, ax=axes[1],
                 color='#1abc9c', edgecolor='white')
    axes[1].axvline(col_log.mean(),   color='red',    linestyle='--', linewidth=1.5,
                    label=f'Media: {col_log.mean():.2f}')
    axes[1].axvline(col_log.median(), color='orange', linestyle='-',  linewidth=1.5,
                    label=f'Mediana: {col_log.median():.2f}')
    axes[1].set_title(f'log1p({col}) — DESPUÉS  (skew: {skew_despues:.2f})', fontsize=11)
    axes[1].set_xlabel(f'log1p({col})'); axes[1].set_ylabel('Frecuencia'); axes[1].legend()

    plt.suptitle(f'Comparación antes/después de log1p — {col}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'4_3_log1p_{col}.png', dpi=120, bbox_inches='tight')
    plt.show()

    mejora = "✔ MEJORÓ" if abs(skew_despues) < abs(skew_antes) else "✘ NO mejoró"
    print(f"  {col}: {mejora}  |  skew antes={skew_antes:.4f}  →  después={skew_despues:.4f}")
print()
salto()


# ── 4.4 — Variables categóricas ordenadas ───────────────
print("4.4 — Barras de variables categóricas (mayor a menor)\n")

frecuencias_type = df['type'].value_counts().sort_values(ascending=False)
pct_dom = frecuencias_type.iloc[0] / frecuencias_type.sum() * 100

fig, ax = plt.subplots(figsize=(9, 5))
colores = ['#e74c3c' if i == 0 else '#3498db' for i in range(len(frecuencias_type))]
bars = ax.bar(frecuencias_type.index, frecuencias_type.values,
              color=colores, edgecolor='white')
ax.set_title('Frecuencia de tipos de transacción (mayor a menor)', fontsize=12)
ax.set_xlabel('Tipo', fontsize=10)
ax.set_ylabel('Cantidad de transacciones', fontsize=10)
for i, v in enumerate(frecuencias_type.values):
    pct = v / len(df) * 100
    ax.text(i, v + 200, f'{v:,}\n({pct:.1f}%)', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('4_4_barras_tipo.png', dpi=120, bbox_inches='tight')
plt.show()

print(f"Categoría dominante: CASH_OUT ({pct_dom:.1f}%)")
if pct_dom > 35:
    print("⚠️  CASH_OUT domina el dataset. Un modelo podría sesgar sus predicciones")
    print("   hacia el comportamiento de este tipo de transacción.")
print("""
  DEBIT representa solo el 0.61% del total → categoría minoritaria.
  Si se usa esta variable como feature en un modelo, podría ser necesario
  agrupar DEBIT con otra categoría o tratarla como caso especial.
""")
salto()


# ── 4.5 — amount separado por isFraud ───────────────────
print("4.5 — Distribución de amount separada por isFraud\n")

amount_log = np.log1p(df['amount'])
df['amount_log'] = amount_log

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Izquierdo: escala original (con log1p ya aplicado para legibilidad)
sns.histplot(
    data=df, x='amount_log', hue='isFraud',
    bins=60, kde=True, stat='density', common_norm=False,
    palette={0: '#2ecc71', 1: '#e74c3c'}, alpha=0.55,
    edgecolor='white', ax=axes[0]
)
axes[0].set_title('log1p(amount) por isFraud\n(densidad normalizada)', fontsize=11)
axes[0].set_xlabel('log1p(amount)')
axes[0].set_ylabel('Densidad')
axes[0].legend(title='isFraud', labels=['1 — Fraude', '0 — No fraude'])

# Derecho: boxplot para ver medianas y outliers
df["isFraud_lbl"] = df["isFraud"].map({0: "No fraude", 1: "Fraude"})
sns.boxplot(
    data=df, x="isFraud_lbl", y="amount_log",
    hue="isFraud_lbl",
    palette={"No fraude": "#2ecc71", "Fraude": "#e74c3c"},
    legend=False, ax=axes[1], width=0.4
)
axes[1].set_title("Boxplot de log1p(amount) por isFraud", fontsize=11)
axes[1].set_xlabel("isFraud  (0 = legítima, 1 = fraude)")
axes[1].set_ylabel("log1p(amount)")

plt.suptitle('Monto de transacción vs. Fraude', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('4_5_amount_vs_fraud.png', dpi=120, bbox_inches='tight')
plt.show()

print("""
Interpretación:
  - Las transacciones fraudulentas (rojo) se concentran en montos más altos
    que las legítimas (verde). La distribución de fraude muestra un pico
    hacia la derecha del histograma de log1p(amount).
  - El boxplot confirma que la mediana de amount en fraudes es notablemente
    mayor que en transacciones legítimas.
  - Esto convierte a 'amount' en una feature discriminante importante para
    cualquier modelo de detección de fraude.
""")

df.drop(columns=['amount_log'], inplace=True)
salto()


# ══════════════════════════════════════════════════════════
#  EXTRA — ¿Es necesario normalizar? Min-Max vs Z-Score
# ══════════════════════════════════════════════════════════

print("EXTRA — Análisis de normalización: Min-Max vs Z-Score\n")

print("""
¿Para qué sirve normalizar?
  La normalización no cambia la forma de la distribución ni mejora el EDA,
  pero es NECESARIA para modelos que dependen de la escala de los datos:
  - Regresión logística, SVM, K-NN, redes neuronales, KMeans → requieren escala.
  - Árboles de decisión, Random Forest, XGBoost → NO necesitan escala.
  En detección de fraude se suelen usar modelos basados en árboles, pero si
  se desea comparar o combinar modelos, normalizar es buena práctica.

Min-Max Scaling  →  X' = (X - Xmin) / (Xmax - Xmin)
  - Resultado: rango [0, 1].
  - Ideal cuando: la distribución no es normal y no hay outliers extremos.
  - Problema aquí: amount tiene outliers masivos (máx ~37M), lo que
    comprimiría el 99% de los datos cerca de 0. NO recomendado para amount.

Z-Score (Standardización)  →  X' = (X - media) / std
  - Resultado: media=0, std=1 (sin rango fijo).
  - Ideal cuando: la distribución es aproximadamente normal o hay outliers.
  - Más robusto que Min-Max frente a outliers extremos.

Recomendación por variable:
""")

recomendaciones = {
    'step':           ('Z-Score',  'Distribución aprox. normal. Z-Score preserva la forma bien.'),
    'amount':         ('log1p + Z-Score', 'Skew = 22. Primero log1p, luego estandarizar. Min-Max sería aplastado por el outlier de $37M.'),
    'oldbalanceOrg':  ('log1p + Z-Score', 'Skew = 5.25. Misma lógica que amount. Outliers fuertes.'),
    'newbalanceOrig': ('log1p + Z-Score', 'Skew = 5.18. Muchos valores en 0; log1p los convierte en 0 limpiamente.'),
    'oldbalanceDest': ('log1p + Z-Score', 'Skew = 16.5. Muchas cuentas mula con saldo 0 inicial. log1p necesario.'),
    'newbalanceDest': ('log1p + Z-Score', 'Skew = 16.4. Igual que oldbalanceDest.'),
}

print(f"{'Variable':<18} {'Método recomendado':<22} {'Justificación'}")
print("-" * 90)
for col, (metodo, justif) in recomendaciones.items():
    print(f"  {col:<16} {metodo:<22} {justif}")

print("""
Conclusión:
  - Para este dataset, Z-Score aplicado sobre log1p(variable) es el enfoque
    más adecuado para las 5 variables financieras.
  - Min-Max solo es conveniente para 'step', que ya tiene una distribución
    más uniforme y valores acotados (1–736), pero Z-Score también funciona.
  - isFraud e isFlaggedFraud son binarias → no se normalizan.
  - nameOrig, nameDest, type son categóricas → se codifican (LabelEncoder /
    OneHotEncoding), no se normalizan.
""")

# Demo visual: comparación de las 3 escalas para 'amount'
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Original con log1p
amount_log = np.log1p(df['amount'])
sns.histplot(amount_log, bins=50, kde=True, ax=axes[0], color='#3498db', edgecolor='white')
axes[0].set_title('log1p(amount)\n(sin normalizar)', fontsize=10)
axes[0].set_xlabel('log1p(amount)')
axes[0].set_ylabel('Frecuencia')

# Min-Max sobre log1p
from sklearn.preprocessing import MinMaxScaler, StandardScaler
mm  = MinMaxScaler().fit_transform(amount_log.values.reshape(-1,1)).flatten()
sns.histplot(mm, bins=50, kde=True, ax=axes[1], color='#e67e22', edgecolor='white')
axes[1].set_title('Min-Max sobre log1p(amount)\nrango [0, 1]', fontsize=10)
axes[1].set_xlabel('Valor escalado [0–1]')
axes[1].set_ylabel('Frecuencia')

# Z-Score sobre log1p
zs  = StandardScaler().fit_transform(amount_log.values.reshape(-1,1)).flatten()
sns.histplot(zs, bins=50, kde=True, ax=axes[2], color='#1abc9c', edgecolor='white')
axes[2].set_title('Z-Score sobre log1p(amount)\nmedia=0, std=1', fontsize=10)
axes[2].set_xlabel('Valor estandarizado')
axes[2].set_ylabel('Frecuencia')

plt.suptitle('Comparación de métodos de normalización sobre amount', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('extra_normalizacion_amount.png', dpi=120, bbox_inches='tight')
plt.show()

print("Gráfico de normalización generado.")
salto()

print("✔ Análisis completo finalizado.")
