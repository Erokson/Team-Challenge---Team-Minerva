
# Nombre del módulo: toolbox_ML.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, f_oneway, kruskal


# 1) describe_df

def describe_df(df):
    """
    Devuelve un DataFrame con la descripción de cada columna del DataFrame de entrada.
    Muestra:
       - Tipo de dato
       - Porcentaje de valores nulos
       - Número de valores únicos
       - Porcentaje de cardinalidad

    Argumentos:
    df (pd.DataFrame): El DataFrame a describir.

    Retorna:
    pd.DataFrame: DataFrame con la información solicitada para cada columna.
    """

    # Listas para ir almacenando la información
    col_names = []
    dtypes = []
    perc_nulls = []
    unique_vals = []
    perc_cardinality = []

    n_rows = len(df)

    for col in df.columns:
        col_names.append(col)
        dtypes.append(df[col].dtype)

        # Porcentaje de nulos
        null_count = df[col].isna().sum()
        perc_null = (null_count / n_rows) * 100 if n_rows else 0
        perc_nulls.append(round(perc_null, 2))

        # Número de valores únicos
        nunique = df[col].nunique(dropna=False)
        unique_vals.append(nunique)

        # Porcentaje de cardinalidad
        perc_card = (nunique / n_rows) * 100 if n_rows else 0
        perc_cardinality.append(round(perc_card, 2))

    # Construimos el DataFrame de salida
    summary_df = pd.DataFrame({
        'columna': col_names,
        'tipo': dtypes,
        '%_nulos': perc_nulls,
        'valores_unicos': unique_vals,
        '%_cardinalidad': perc_cardinality
    })

    return summary_df


# 2) tipifica_variables
def tipifica_variables(df, umbral_categoria=10, umbral_continua=0.2):
    """
    Asigna un tipo sugerido (Binaria, Categórica, Numérica Discreta o Numérica Continua)
    en función de la cardinalidad de cada columna del DataFrame.

    Argumentos:
    df (pd.DataFrame): El DataFrame a analizar.
    umbral_categoria (int): Umbral máximo de cardinalidad para considerar una variable como categórica.
    umbral_continua (float): Umbral (proporción) de cardinalidad por encima de la cual
                             se considera una variable como numérica continua.

    Retorna:
    pd.DataFrame: DataFrame con dos columnas: "nombre_variable" y "tipo_sugerido".
    """

    resultados = {
        'nombre_variable': [],
        'tipo_sugerido': []
    }

    n_rows = len(df)

    for col in df.columns:
        # Cardinalidad absoluta
        card_abs = df[col].nunique(dropna=False)
        # Cardinalidad relativa
        card_rel = card_abs / n_rows if n_rows else 0

        # Lógica de clasificación
        if card_abs == 2:
            tipo = 'Binaria'
        elif card_abs < umbral_categoria:
            tipo = 'Categórica'
        else:
            if card_rel >= umbral_continua:
                tipo = 'Numerica Continua'
            else:
                tipo = 'Numerica Discreta'

        resultados['nombre_variable'].append(col)
        resultados['tipo_sugerido'].append(tipo)

    return pd.DataFrame(resultados)



# 3) get_features_num_regression
# Nueva implementación

def get_features_num_regression(df, target_col, umbral_corr, pvalue=None, tipo_variables=None ):
    """
    Selecciona características numéricas que tengan una correlación significativa
    con la variable objetivo.

    Parámetros:
    df (pd.DataFrame): DataFrame que contiene las variables.
    target_col (str): Nombre de la columna objetivo.
    umbral_corr (float): Umbral mínimo de correlación (valor absoluto) para seleccionar las variables.
    pvalue (float, opcional): Valor p para filtrar variables con significancia estadística.
                              Por defecto es None.

    Retorna:
    list: Lista de nombres de columnas numéricas correlacionadas con el target,
          o None si los parámetros son inválidos.
    """
    # Validaciones de entrada
    if not isinstance(df, pd.DataFrame):
        print("El argumento 'df' debe ser un DataFrame.")
        return None

    if target_col not in df.columns:
        print(f"La columna objetivo '{target_col}' no se encuentra en el DataFrame.")
        return None

    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"La columna objetivo '{target_col}' debe ser numérica.")
        return None

    if not (0 <= umbral_corr <= 1):
        print("El valor de 'umbral_corr' debe estar entre 0 y 1.")
        return None

    if pvalue is not None and not (0 <= pvalue <= 1):
        print("El valor de 'pvalue' debe estar entre 0 y 1.")
        return None
    
    if tipo_variables is None or 'nombre_variable' not in tipo_variables.columns or 'tipo_sugerido' not in tipo_variables.columns:
        print("El argumento 'tipo_variables' debe ser un DataFrame generado por la función tipifica_variables.")
        return None
    
     # Inicializar lista para las columnas seleccionadas
    selected_features = []

    # Selección de columnas numéricas según la clasificación
    valid_types = ['Numerica Continua', 'Numerica Discreta']
    valid_cols = tipo_variables[tipo_variables['tipo_sugerido'].isin(valid_types)]['nombre_variable'].tolist()

    if target_col in valid_cols:
        valid_cols.remove(target_col)  # Excluir la columna objetivo

    for col in valid_cols:
        # Correlación (método Pearson simplificado usando .corr)
        corr = df[target_col].corr(df[col])

        if abs(corr) >= umbral_corr:
            if pvalue is not None:
                # Test de significancia estadística con pearsonr
                _, p_val = pearsonr(df[target_col], df[col])
                if p_val < pvalue:
                    selected_features.append(col)
            else:
                selected_features.append(col)

    return selected_features




# 4) plot_features_num_regression
# Nueva implementación

def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
    Genera gráficos pairplot para las variables numéricas seleccionadas
    frente a la variable objetivo.

    Parámetros:
    df (pd.DataFrame): DataFrame con los datos.
    target_col (str, opcional): Nombre de la columna objetivo. Por defecto es una cadena vacía.
    columns (list, opcional): Lista de columnas numéricas a evaluar.
                              Por defecto es una lista vacía.
    umbral_corr (float, opcional): Umbral mínimo de correlación. Por defecto es 0.
    pvalue (float, opcional): Nivel de significancia estadística para el test de correlación.
                              Por defecto es None.

    Retorna:
    list: Lista de columnas que cumplen con las condiciones de correlación y significancia.
    """
    # Validaciones de entrada
    if not isinstance(df, pd.DataFrame):
        print("El argumento 'df' debe ser un DataFrame.")
        return None

    if target_col not in df.columns:
        print(f"La columna objetivo '{target_col}' no se encuentra en el DataFrame.")
        return None

    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"La columna objetivo '{target_col}' debe ser numérica.")
        return None

    if not (0 <= umbral_corr <= 1):
        print("El valor de 'umbral_corr' debe estar entre 0 y 1.")
        return None

    if pvalue is not None and not (0 <= pvalue <= 1):
        print("El valor de 'pvalue' debe estar entre 0 y 1.")
        return None

    # Si no se proporcionan columnas, seleccionamos todas las columnas numéricas 
    # excepto la columna objetivo
    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns.drop(target_col).tolist()

    selected_features = []
    for col in columns:
        corr = df[target_col].corr(df[col])
        if abs(corr) >= umbral_corr:
            if pvalue is not None:
                _, p_val = pearsonr(df[target_col], df[col])
                if p_val < pvalue:
                    selected_features.append(col)
            else:
                selected_features.append(col)

    # Generación de gráficos pairplot en lotes de hasta 5 columnas
    max_features = 5
    for i in range(0, len(selected_features), max_features):
        subset = [target_col] + selected_features[i:i + max_features]
        sns.pairplot(df[subset])
        plt.show()

    return selected_features


def plot_features_cat_regression(dataframe, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
    
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("El argumento 'dataframe' debe ser un DataFrame.")
    if target_col not in dataframe.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no está en el DataFrame.")
    if columns and not all(col in dataframe.columns for col in columns):
        raise ValueError("Una o más columnas en 'columns' no están en el DataFrame.")
    if not (0 <= pvalue <= 1):
        raise ValueError("El valor de 'pvalue' debe estar entre 0 y 1.")

 
    selected_columns = []


    if columns:
        for col in columns:
            if dataframe[col].dtype == 'object':  # Verificar si es categórica
                # Realizar test estadístico (por ejemplo, ANOVA o Kruskal-Wallis)
                stat, p_val = realizar_prueba_estadistica(dataframe[col], dataframe[target_col])
                if p_val < pvalue:
                    selected_columns.append(col)

               
                if with_individual_plot:
                    graficar_histograma(dataframe[col], dataframe[target_col])


    else:
        
        numeric_columns = dataframe.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            stat, p_val = realizar_prueba_estadistica(dataframe[col], dataframe[target_col])
            if p_val < pvalue:
                selected_columns.append(col)

            
            if with_individual_plot:
                graficar_histograma(dataframe[col], dataframe[target_col])

    
    return selected_columns




