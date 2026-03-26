import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




# ==========================================================
# Tabla y gráfico de frecuencias
# ==========================================================
def freq(df, v, plot=True):
    """
    Genera tabla de frecuencias para una variable categórica o discreta.
    Pensada como función de exploración (EDA), similar a profiling.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene la variable.
    v : str
        Nombre de la variable a analizar.
    plot : bool, default=True
        Si es True, muestra gráfico de barras con frecuencia absoluta.

    Retorna
    -------
    None
        Imprime tabla de frecuencias y porcentaje de missings.
    """

    # Validación de existencia
    if v not in df.columns:
        raise ValueError(f"La variable '{v}' no existe en el DataFrame.")

    # Cálculo de frecuencias
    aux = df[v].value_counts(dropna=False).to_frame(name='FA')

    total_valid = aux['FA'].sum()

    if total_valid == 0:
        print(f"La variable '{v}' no tiene valores válidos.")
        return
    
    aux['FR'] = aux['FA'] / total_valid
    aux['FAA'] = aux['FA'].cumsum()
    aux['FRA'] = aux['FR'].cumsum()

    # Impresión de resultados
    print("\n\n\n", '{:^65}'.format(v), "\n\n")
    print(aux, "\n")

    # Gráfico de barras (opcional)
    if plot:
        aux_plot = aux.reset_index()

        plt.figure(figsize=(8, 4))
        plt.bar(aux_plot.iloc[:, 0].astype(str), aux_plot['FA'])
        plt.xticks(rotation=45)
        plt.title(f'Frecuencia - {v}')
        plt.tight_layout()
        plt.show()




# ==========================================================
# Valores Extremos
# ==========================================================
def ext_val(df, var, method='iqr', k=1.5, p_low=0.01, p_high=0.99):
    """
    Marca valores extremos en una variable numérica.

    Parámetros
    ----------
    df : pandas.DataFrame
    var : str
        Nombre de la variable numérica.
    method : str
        'iqr' o 'percentile'
    k : float
        Multiplicador para IQR (default 1.5 clásico)
    p_low : float
        Percentil inferior si method='percentile'
    p_high : float
        Percentil superior si method='percentile'

    Retorna
    -------
    DataFrame con columna adicional: 'flag_ext_<var>'
    """

    # Validaciones
    if var not in df.columns:
        raise ValueError(f"La variable '{var}' no existe en el DataFrame.")

    if not np.issubdtype(df[var].dtype, np.number):
        raise TypeError(f"La variable '{var}' debe ser numérica.")

    serie = df[var].dropna()

    if len(serie) == 0:
        print(f"La variable '{var}' no tiene valores numéricos válidos.")
        return df

    # Cálculo de límites
    if method == 'iqr':
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        li = q1 - k * iqr
        ls = q3 + k * iqr

    elif method == 'percentile':
        li = serie.quantile(p_low)
        ls = serie.quantile(p_high)

    else:
        raise ValueError("method debe ser 'iqr' o 'percentile'.")

    # Creación del flag
    flag_name = f'flag_ext_{var}'
    df[flag_name] = ((df[var] < li) | (df[var] > ls)).astype(int)

    # Información de impacto
    total = len(df)
    n_flag = df[flag_name].sum()

    print("\n" + "="*60)
    print(f"Variable: {var}")
    print(f"Método: {method}")
    print(f"Límite inferior: {li:.4f}")
    print(f"Límite superior: {ls:.4f}")
    print(f"Registros marcados: {n_flag} ({n_flag/total:.2%})")
    print("="*60 + "\n")

    return df