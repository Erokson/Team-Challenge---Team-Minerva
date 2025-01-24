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
