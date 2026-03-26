import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer




# ==========================================================
# IV para variables sin binarizar
# ==========================================================
def iv_screening(df, var, tgt, n_bins=5,type_var='cat'):

    """
    Calcula IV para una sola variable y dependiendo de su tipo
    la normaliza o discretiza.

    Parámetros
    ----------
    df : DataFrame
    var : str
        Nombre de la variable
    tgt : str
        Variable objetivo binaria (0/1)
    n_bins : int
        Número de bins para variables numéricas (default 5)
    type_var : str
        'cat' o 'num' (default 'cat') para indicar el tipo de variable.

    Retorna
    -------
    Número con el IV de la variable. Para variables numéricas se discretiza en n_bins usando pd.qcut. 
    Para variables categóricas se agrupa por categoría o se agrupan en bins según su tasa de malos. 
    """

    serie = df[var]

    # NUMÉRICA
    if np.issubdtype(serie.dtype, np.number) and type_var != 'cat':

        bins = pd.qcut(
            serie,
            q=n_bins,
            duplicates="drop"
        )

        aux = pd.DataFrame({
            "bin": bins,
            tgt: df[tgt]
        })

    
    # CATEGÓRICA
    else:

        aux = df[[var, tgt]].copy()
        aux[var] = aux[var].fillna("missing")

       
        cat_rate = (
            aux.groupby(var)[tgt]
            .agg(["count", "mean"])
            .reset_index()
        )

        cat_rate.columns = [var, "Total", "Tasa_malos"]
        cat_rate = cat_rate.sort_values("Tasa_malos")

        # ranking para poder binarizar rankenado de forma pareja los empates (mismo riesgo → mismo bin)
        cat_rate["rank"] = cat_rate["Tasa_malos"].rank(method="dense")

        # Caso 1: pocos niveles → cada nivel es un bin
        if len(cat_rate['rank'].unique()) <= n_bins:
            cat_rate["bin"] = cat_rate["rank"]

        # Caso 2: demasiados niveles → agrupar con qcut
        else:
            cat_rate["bin"] = pd.qcut(
                cat_rate["rank"],
                q=n_bins,
                duplicates="drop"
            )

        mapping = dict(zip(cat_rate[var], cat_rate["bin"]))

        aux["bin"] = aux[var].map(mapping)

    
    # tabla base
    tab = (
        aux.groupby("bin")[tgt]
        .agg(["count", "sum"])
        .reset_index()
    )

    tab.columns = ["bin", "Total", "Malos"]

    tab["Buenos"] = tab["Total"] - tab["Malos"]

    total_obs = tab["Total"].sum()

    
    # recalcular totales
    total_malos = tab["Malos"].sum()
    total_buenos = tab["Buenos"].sum()

    tab["% Malos"] = tab["Malos"] / total_malos
    tab["% Buenos"] = tab["Buenos"] / total_buenos

    
    # WoE
    smoothing=0.5
    tab["WoE"] = np.log(
        ((tab["Buenos"] + smoothing) / total_buenos) /
        ((tab["Malos"] + smoothing) / total_malos)
    )

    
    # IV
    tab["IV"] = (tab["% Buenos"] - tab["% Malos"]) * tab["WoE"]

    return tab["IV"].sum()




# ==========================================================
# IV para variables ya binarizadas
# ==========================================================
def iv_binned(df, bin_vars, tgt):
    """
    Calcula IV para una lista de variables binarizadas.

    Parámetros
    ----------
    df : DataFrame
    bin_vars : list
        Lista de variables ya binarizadas (ej: ['BIN_edad', 'BIN_ingreso'])
    tgt : str
        Variable objetivo binaria (0/1)

    Retorna
    -------
    DataFrame ordenado por IV descendente
    """

    results = []

    for var in bin_vars:

        tab = (df[[var, tgt]]
            .assign(**{var: df[var].astype(str)})
            .fillna('missing')
            .groupby(var)[tgt]
            .agg(['count', 'sum'])
            .reset_index()
        )

        tab.columns = [var, 'Total', 'Malos']
        tab['Buenos'] = tab['Total'] - tab['Malos']

        total_malos = tab['Malos'].sum()
        total_buenos = tab['Buenos'].sum()

        # Evitar división por cero
        if total_malos == 0 or total_buenos == 0:
            iv = 0
        else:
            tab['% Malos'] = tab['Malos'] / total_malos
            tab['% Buenos'] = tab['Buenos'] / total_buenos

            # Smoothing pequeño para evitar log(0)
            eps = 0.5
            tab['WoE'] = np.log(((tab['Buenos'] + eps) / total_buenos) /((tab['Malos'] + eps) / total_malos))

            tab['IV_bin'] = (tab['% Buenos'] - tab['% Malos']) * tab['WoE']
            iv = tab['IV_bin'].sum()

        results.append({'Variable': var,'IV': iv})

    iv_df = pd.DataFrame(results)
    iv_df.sort_values('IV', ascending=False, inplace=True)
    iv_df.reset_index(drop=True, inplace=True)
    iv_df['Interpretación'] = iv_df['IV'].apply(lambda x: 'Muy Predictiva' if x > 0.3 else ('Predictiva' if x > 0.1 else ('Débil' if x > 0.02 else 'No Predictiva')))


    return iv_df




# ==========================================================
# Binararización manual de variables categóricas
# ==========================================================
def binning_cat(df, columns, tgt):

    """
    Genera tabla de binning para variables categóricas.
    Pensada para análisis manual de riesgo y exportación a Excel.

    Parámetros
    ----------
    df : DataFrame
    columns : list
        Lista de variables categóricas a binarizar (ej: ['OCUPACION', 'NIVEL_EDUCATIVO'])
    tgt : str
        Variable objetivo binaria (0/1)

    Retorna
    -------
    DataFrame con columnas: [columns..., 'Total', 'Malos', 'Buenos', 'Tasa de Malos', '% Total', '% Malos', '% Buenos', 'WoE', 'IV']
    """

    # Validaciones
    if tgt not in df.columns:
        raise ValueError(f"Target '{tgt}' no existe en el DataFrame.")

    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Variable '{col}' no existe en el DataFrame.")

    # Validar target binario
    if df[tgt].nunique() != 2:
        raise ValueError("El target debe ser binario (0/1).")

    # Agrupación
    bining = (
        df[[tgt] + columns]
        .assign(**{col: df[col].astype(str) for col in columns})
        .fillna('missing')
        .groupby(columns)[tgt]
        .agg(['count', 'sum'])
        .reset_index()
    )

    bining.columns = columns + ['Total', 'Malos']
    bining['Buenos'] = bining['Total'] - bining['Malos']
    bining['Tasa de Malos'] = bining['Malos'] / bining['Total']

    total_malos = bining['Malos'].sum()
    total_buenos = bining['Buenos'].sum()
    total_total = bining['Total'].sum()
    
    bining['% Total'] = bining['Total'] / total_total
    bining['% Malos'] = bining['Malos'] / total_malos
    bining['% Buenos'] = bining['Buenos'] / total_buenos

    # IV y WoE con smoothing
    smoothing = 0.5
    bining['WoE'] = np.log(((bining['Buenos'] + smoothing) / total_buenos) /((bining['Malos'] + smoothing) / total_malos))

    bining['IV'] = (bining['% Buenos'] - bining['% Malos']) * bining['WoE']
    bining = bining.sort_values('Tasa de Malos', ascending=True)

    return bining




# ==========================================================
# Binararización manual de variables numéricas
# ==========================================================
def binning_cont(df, var, tgt, n_bins=5, strategy='quantile'):
    """
    Discretiza variable numérica en n_bins y genera tabla de riesgo.
    Siempre incluye NA como categoría 'missing'.

    Parámetros
    ----------
    df : DataFrame
    var : str
        Variable numérica a discretizar
    tgt : str
        Variable objetivo binaria (0/1)
    n_bins : int
        Número de bins (default 5)
    strategy : str
        'uniform', 'quantile' o 'kmeans' (default 'quantile')  
    
    Retorna
    -------
    DataFrame con columnas: ['BIN_<var>', 'Total', 'Malos', 'Buenos', 'Tasa de Malos', '% Total', '% Malos', '% Buenos', 'WoE', 'IV']
    """

    # Validaciones mínimas necesarias
    if var not in df.columns:
        raise ValueError(f"La variable '{var}' no existe.")

    if tgt not in df.columns:
        raise ValueError(f"El target '{tgt}' no existe.")

    if not np.issubdtype(df[var].dtype, np.number):
        raise TypeError(f"La variable '{var}' debe ser numérica.")

    # Crear bins numéricos (sin NA)
    aux = df[[var, tgt]].copy()
    mask_notna = aux[var].notna()

    kb = KBinsDiscretizer(
        n_bins=n_bins,
        encode='ordinal',
        strategy=strategy,
        random_state=123
    )

    kb.fit(aux.loc[mask_notna, [var]])

    bins = kb.bin_edges_[0]
    bins[0] = aux[var].min() - 1  # Asegurar que el mínimo quede incluido
    bins[-1] = aux[var].max() + 1  # Asegurar que el máximo quede incluido

    aux.loc[mask_notna, f'BIN_{var}'] = pd.cut(
        aux.loc[mask_notna, var],
        bins=bins,
        include_lowest=True,
        right=True
    ).astype(str)

    # Asignar missing explícito
    aux.loc[~mask_notna, f'BIN_{var}'] = 'missing'

    bining = (
        aux
        .groupby(f'BIN_{var}')[tgt]
        .agg(['count', 'sum'])
        .reset_index()
    )

    bining.columns = [f'BIN_{var}', 'Total', 'Malos']
    bining['Buenos'] = bining['Total'] - bining['Malos']
    bining['Tasa de Malos'] = bining['Malos'] / bining['Total']

    total_malos = bining['Malos'].sum()
    total_buenos = bining['Buenos'].sum()
    total_total = bining['Total'].sum()
    
    bining['% Total'] = bining['Total'] / total_total
    bining['% Malos'] = bining['Malos'] / total_malos
    bining['% Buenos'] = bining['Buenos'] / total_buenos

    # IV y WoE con smoothing
    smoothing = 0.5
    bining['WoE'] = np.log(((bining['Buenos'] + smoothing) / total_buenos) /((bining['Malos'] + smoothing) / total_malos))

    bining['IV'] = (bining['% Buenos'] - bining['% Malos']) * bining['WoE']
    bining = bining.sort_values('Tasa de Malos', ascending=False)

    tab_missing = bining[bining[f'BIN_{var}'] == 'missing']
    tab_others = bining[bining[f'BIN_{var}'] != 'missing'].copy()

    tab_others['orden'] = (
        tab_others[f'BIN_{var}']
        .str.extract(r'\(([^,]+),')[0]
        .astype(float)
    )

    tab_others = tab_others.sort_values('orden')
    tab_others = tab_others.drop(columns='orden')

    # Concatenar con missing arriba
    tab_final = pd.concat([tab_missing, tab_others], axis=0)

    return tab_final




# ==========================================================
# Calcular WoE para variable ya binarizada.
# ==========================================================
"""
Calcula el WoE para una variable ya binarizada, añadiendo la columna 'WOE_<var>' al DataFrame original.
Parámetos
----------
train_df : DataFrame
    DataFrame de entrenamiento
test_df : DataFrame
    DataFrame de test (se le asigna el mismo WoE que train)
var_bin : str
    Nombre de la variable ya binarizada (ej: 'BIN_edad')
tgt : str
    Variable objetivo binaria (0/1)

Retorna
-------
DataFrame de entrenamiento y test con nueva columna 'WOE_<var>'
"""
def calc_woe(train_df, test_df, var_bin, tgt):

    aux = (
        train_df[[var_bin, tgt]]
        .assign(**{var_bin: train_df[var_bin].astype(str)})
        .fillna("missing")
        .groupby(var_bin)[tgt]
        .agg(["count", "sum"])
        .reset_index()
    )

    aux.columns = [var_bin, "Total", "Malos"]
    aux["Buenos"] = aux["Total"] - aux["Malos"]

    total_malos = aux["Malos"].sum()
    total_buenos = aux["Buenos"].sum()

    smoothing=0.5
    aux["WoE"] = np.log(
        ((aux["Buenos"] + smoothing) / total_buenos) /
        ((aux["Malos"] + smoothing) / total_malos)
    )

    woe_map = dict(zip(aux[var_bin], aux["WoE"]))

    var_name = var_bin.replace("BIN_", "")
    train_df[f"WOE_{var_name}"] = train_df[var_bin].map(woe_map).astype(float)
    test_df[f"WOE_{var_name}"] = test_df[var_bin].map(woe_map).astype(float)

    return train_df, test_df