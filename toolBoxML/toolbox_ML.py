def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    """
    Selecciona características numéricas que tengan una correlación significativa con la variable objetivo.

    Parámetros:
    df (pd.DataFrame): DataFrame que contiene las variables.
    target_col (str): Nombre de la columna objetivo.
    umbral_corr (float): Umbral mínimo de correlación (valor absoluto) para seleccionar las variables.
    pvalue (float, opcional): Valor p para filtrar variables con significancia estadística. Por defecto es None.

    Retorna:
    list: Lista de nombres de columnas numéricas correlacionadas con el target, o None si los parámetros son inválidos.
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

    # Selección de columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col)
    selected_features = []

    for col in numeric_cols:
        corr = df[target_col].corr(df[col])
        if abs(corr) >= umbral_corr:
            if pvalue is not None:
                _, p_val = pearsonr(df[target_col], df[col])
                if p_val < pvalue:
                    selected_features.append(col)
            else:
                selected_features.append(col)

    return selected_features

# Función: plot_features_num_regression
def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
    Genera gráficos pairplot para las variables numéricas seleccionadas frente a la variable objetivo.

    Parámetros:
    df (pd.DataFrame): DataFrame con los datos.
    target_col (str, opcional): Nombre de la columna objetivo. Por defecto es una cadena vacía.
    columns (list, opcional): Lista de columnas numéricas a evaluar. Por defecto es una lista vacía.
    umbral_corr (float, opcional): Umbral mínimo de correlación. Por defecto es 0.
    pvalue (float, opcional): Nivel de significancia estadística para el test de correlación. Por defecto es None.

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

    # Selección de columnas numéricas
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

    # Generación de gráficos pairplot
    max_features = 5  # Número máximo de columnas por gráfico
    for i in range(0, len(selected_features), max_features):
        subset = [target_col] + selected_features[i:i + max_features]
        sns.pairplot(df[subset])
        plt.show()

    return selected_features