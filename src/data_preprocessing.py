import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Carrega o dataset de doenças cardíacas
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, target_column='heart_disease'):
    """
    Preprocessa os dados para treinamento
    """
    # Separar features e target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Escalar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler