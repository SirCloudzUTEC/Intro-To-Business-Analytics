import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)

DATA_PATH = "BD_fraud.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50

C = {
    "navy": "#0B1F3A",
    "blue": "#1F4E79",
    "blue_dark": "#153B5C",
    "cyan": "#2A9DCE",
    "teal": "#2A9D8F",
    "emerald": "#1B8A5A",
    "green": "#4CAF50",
    "gold": "#C9A227",
    "orange": "#E76F51",
    "burgundy": "#B23A48",
    "red": "#B23A48",
    "gray_dark": "#3A3A3A",
    "gray": "#8A8F98",
    "gray_light": "#E6E8EB",
    "white": "#FFFFFF",
    "bg": "#F7F9FB"
}

def clean_ax(ax, axis="both"):
    ax.set_facecolor(C["bg"])

    if axis in ["x", "both"]:
        ax.grid(axis="x", alpha=0.18, linewidth=0.8)

    if axis in ["y", "both"]:
        ax.grid(axis="y", alpha=0.18, linewidth=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C["gray_light"])
    ax.spines["bottom"].set_color(C["gray_light"])

    ax.tick_params(axis="both", colors=C["gray_dark"], labelsize=9)


def add_labels(ax, bars, labels, horizontal=False, fs=9):
    for bar, label in zip(bars, labels):
        if horizontal:
            width = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2

            ax.text(
                width + 0.015,
                y,
                label,
                va="center",
                ha="left",
                fontsize=fs,
                fontweight="bold",
                color=C["gray_dark"]
            )
        else:
            height = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2

            ax.text(
                x,
                height,
                label,
                va="bottom",
                ha="center",
                fontsize=fs,
                fontweight="bold",
                color=C["gray_dark"]
            )


df = pd.read_csv(DATA_PATH)

print("\nPrimeras filas del dataset:")
print(df.head())

print("\nDimensiones del dataset:")
print(f"{df.shape[0]:,} filas x {df.shape[1]:,} columnas")

required_cols = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud"
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Faltan columnas en el dataset: {missing_cols}")


#seleccion de target
df["isFraud"] = df["isFraud"].astype(int)

total_fraudes = df["isFraud"].sum()
total_registros = len(df)
fraud_rate = total_fraudes / total_registros

print("\nDistribución real de fraude:")
print(df["isFraud"].value_counts())

print("\nPorcentaje real de fraude:")
print(f"{fraud_rate:.4%}")

print("\nCantidad real de fraudes:")
print(f"{total_fraudes:,}")

# seleccion de variables
features = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

target = "isFraud"

X = df[features].copy()
y = df[target].copy()

categorical_features = ["type"]
numeric_features = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

print("\nVariables usadas como entrada del modelo:")
print(features)

print("\nVariable usada para validar la predicción:")
print(target)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_train_full
)

print("\nDistribución de datos:")
print(f"Train:      {X_train.shape[0]:,} registros | Fraudes: {y_train.sum():,}")
print(f"Validation: {X_val.shape[0]:,} registros | Fraudes: {y_val.sum():,}")
print(f"Test:       {X_test.shape[0]:,} registros | Fraudes: {y_test.sum():,}")


preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numeric_features
        )
    ]
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=8,
    class_weight={0: 1, 1: 25},
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Entrenamiento
print("\nEntrenando modelo ajustado...")

pipeline.fit(X_train, y_train)

print("Modelo entrenado correctamente.")

val_proba = pipeline.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.05, 0.96, 0.01)

threshold_results = []

MIN_PRECISION = 0.60

for threshold in thresholds:
    val_pred = (val_proba >= threshold).astype(int)

    precision_t = precision_score(y_val, val_pred, zero_division=0)
    recall_t = recall_score(y_val, val_pred, zero_division=0)
    f1_t = f1_score(y_val, val_pred, zero_division=0)

    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_val, val_pred).ravel()

    threshold_results.append({
        "threshold": threshold,
        "precision": precision_t,
        "recall": recall_t,
        "f1": f1_t,
        "tn": tn_t,
        "fp": fp_t,
        "fn": fn_t,
        "tp": tp_t
    })

df_thresholds = pd.DataFrame(threshold_results)

valid_thresholds = df_thresholds[
    df_thresholds["precision"] >= MIN_PRECISION
].copy()

if len(valid_thresholds) > 0:
    best_row = valid_thresholds.sort_values(
        ["f1", "recall"],
        ascending=False
    ).iloc[0]
else:
    best_row = df_thresholds.sort_values(
        "f1",
        ascending=False
    ).iloc[0]

BEST_THRESHOLD = best_row["threshold"]

print("\nMejor umbral seleccionado:")
print(f"Threshold:  {BEST_THRESHOLD:.2f}")
print(f"Precision:  {best_row['precision']:.2%}")
print(f"Recall:     {best_row['recall']:.2%}")
print(f"F1-Score:   {best_row['f1']:.2%}")
print(f"FP:         {int(best_row['fp']):,}")
print(f"FN:         {int(best_row['fn']):,}")


y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= BEST_THRESHOLD).astype(int)

df_results = X_test.copy()

df_results["isFraud"] = y_test.values
df_results["FraudProbability"] = y_proba
df_results["NewSystemFlaggedFraud"] = y_pred

df_results["PredictionCorrect"] = (
    df_results["NewSystemFlaggedFraud"] == df_results["isFraud"]
)

df_results["PredictionStatus"] = np.select(
    [
        (df_results["isFraud"] == 0) & (df_results["NewSystemFlaggedFraud"] == 0),
        (df_results["isFraud"] == 0) & (df_results["NewSystemFlaggedFraud"] == 1),
        (df_results["isFraud"] == 1) & (df_results["NewSystemFlaggedFraud"] == 0),
        (df_results["isFraud"] == 1) & (df_results["NewSystemFlaggedFraud"] == 1)
    ],
    [
        "TN - No fraude correctamente ignorado",
        "FP - Falsa alerta de fraude",
        "FN - Fraude no detectado",
        "TP - Fraude correctamente detectado"
    ],
    default="Sin clasificar"
)

df_results.to_csv("Resultados_NewSystemFlaggedFraud.csv", index=False)

print("\nArchivo generado:")
print("Resultados_NewSystemFlaggedFraud.csv")

tn, fp, fn, tp = confusion_matrix(
    df_results["isFraud"],
    df_results["NewSystemFlaggedFraud"]
).ravel()

accuracy = accuracy_score(
    df_results["isFraud"],
    df_results["NewSystemFlaggedFraud"]
)

precision = precision_score(
    df_results["isFraud"],
    df_results["NewSystemFlaggedFraud"],
    zero_division=0
)

recall = recall_score(
    df_results["isFraud"],
    df_results["NewSystemFlaggedFraud"],
    zero_division=0
)

f1 = f1_score(
    df_results["isFraud"],
    df_results["NewSystemFlaggedFraud"],
    zero_division=0
)

miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

roc_auc = roc_auc_score(
    df_results["isFraud"],
    df_results["FraudProbability"]
)

pr_auc = average_precision_score(
    df_results["isFraud"],
    df_results["FraudProbability"]
)

print("\nMatriz de confusión final contra isFraud:")
print(f"TN: {tn:,}")
print(f"FP: {fp:,}")
print(f"FN: {fn:,}")
print(f"TP: {tp:,}")

print("\nMétricas finales del nuevo sistema:")
print(f"Threshold usado: {BEST_THRESHOLD:.2f}")
print(f"Accuracy:       {accuracy:.2%}")
print(f"Precision:      {precision:.2%}")
print(f"Recall:         {recall:.2%}")
print(f"F1-Score:       {f1:.2%}")
print(f"Miss Rate:      {miss_rate:.2%}")
print(f"ROC AUC:        {roc_auc:.2%}")
print(f"PR AUC:         {pr_auc:.2%}")

print("\nReporte de clasificación:")
print(
    classification_report(
        df_results["isFraud"],
        df_results["NewSystemFlaggedFraud"],
        target_names=["No Fraude", "Fraude"]
    )
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 6),
    constrained_layout=True,
    facecolor=C["bg"]
)

fig.suptitle(
    "Efectividad del Sistema Automático de Detección",
    fontsize=16,
    fontweight="bold",
    color=C["navy"]
)

# ── Matriz de Confusión
ax = axes[0]
ax.axis("off")
ax.set_title("Matriz de Confusión", color=C["navy"], pad=14)

cells = [
    (0.08, 0.52, "TN", tn, C["teal"]),
    (0.54, 0.52, "FP", fp, C["gold"]),
    (0.08, 0.12, "FN", fn, C["burgundy"]),
    (0.54, 0.12, "TP", tp, C["emerald"]),
]

for x0, y0, lab, val, col in cells:
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            0.36,
            0.28,
            boxstyle="round,pad=0.02,rounding_size=0.035",
            facecolor=col,
            edgecolor=C["white"],
            linewidth=2,
            transform=ax.transAxes
        )
    )

    ax.text(
        x0 + 0.18,
        y0 + 0.14,
        f"{lab}\n{val:,}",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=C["white"],
        transform=ax.transAxes
    )

ax.text(
    0.5,
    0.88,
    "Predicción",
    ha="center",
    fontsize=9,
    color=C["gray"],
    transform=ax.transAxes
)

ax.text(
    0.26,
    0.82,
    "No fraude",
    ha="center",
    fontsize=8.5,
    color=C["gray_dark"],
    transform=ax.transAxes
)

ax.text(
    0.72,
    0.82,
    "Fraude",
    ha="center",
    fontsize=8.5,
    color=C["gray_dark"],
    transform=ax.transAxes
)

ax.text(
    0.02,
    0.5,
    "Clase real: isFraud",
    ha="center",
    va="center",
    rotation=90,
    fontsize=9,
    color=C["gray"],
    transform=ax.transAxes
)

# ── Métricas principales
ax = axes[1]

metrics = pd.Series({
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
    "Miss Rate": miss_rate
}).sort_values()

metric_colors = [
    C["burgundy"] if i == "Miss Rate" else C["blue_dark"]
    for i in metrics.index
]

bars = ax.barh(
    metrics.index,
    metrics.values,
    color=metric_colors,
    height=0.5
)

ax.set_title("Métricas Principales", color=C["navy"], pad=14)
ax.set_xlim(0, 1.18)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

add_labels(
    ax,
    bars,
    [f"{v:.2%}" for v in metrics.values],
    horizontal=True,
    fs=9.5
)

clean_ax(ax, "x")

plt.savefig(
    "Efectividad_New_System.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=C["bg"]
)

plt.show()


