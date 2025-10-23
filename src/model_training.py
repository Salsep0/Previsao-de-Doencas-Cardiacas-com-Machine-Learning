from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib

def train_logistic_regression(X_train_scaled, y_train):
    """
    Treina modelo de Regressão Logística
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model

def train_random_forest(X_train, y_train):
    """
    Treina modelo de Random Forest
    """
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=None, 
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name, scaled=False):
    """
    Avalia o modelo e retorna métricas
    """
    if scaled:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

def save_model(model, file_path):
    """
    Salva o modelo treinado
    """
    joblib.dump(model, file_path)

def load_model(file_path):
    """
    Carrega um modelo salvo
    """
    return joblib.load(file_path)