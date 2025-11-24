import pandas as pd
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def entrenar_ensemble():
   
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'tickets_procesado.csv')
    model_path = os.path.join(base_dir, 'src', 'modelo_entrenado.pkl')
    vectorizer_path = os.path.join(base_dir, 'src', 'vectorizador.pkl')
    
    print(f" Cargando dataset desde: {input_file}")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(" Error: No se encuentra el CSV.")
        return

    df = df.dropna(subset=['descripcion_limpia', 'categoria'])

    #  TÉCNICA 1 Data Augmentation
  
    print(" Aplicando Ampliación de Datos (x10)...")
    df_ampliado = pd.concat([df] * 10, ignore_index=True)

    X = df_ampliado['descripcion_limpia']
    y = df_ampliado['categoria']

    #  Separar Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

   
   
    print(" Vectorizando (TF-IDF + N-Grams)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    #  ENSEMBLE 
    # Usamos los 3 modelos sugeridos en la consigna 
    print(" Entrenando Comité de Modelos (Naive Bayes + SVM + LogReg)...")
    
    clf1 = MultinomialNB()
    clf2 = SGDClassifier(loss='hinge', random_state=42) # Esto es un SVM lineal rápido
    clf3 = LogisticRegression(random_state=42)

    model = VotingClassifier(estimators=[
        ('nb', clf1), 
        ('svm', clf2), 
        ('lr', clf3)
    ], voting='hard')

    model.fit(X_train_vec, y_train)

    print("\n" + "="*40)
    print(" RESULTADOS DEL ENSEMBLE")
    print("="*40)
    
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    
    print(f" Accuracy (Precisión): {acc:.2f} ({(acc*100):.0f}%)")
    
    if acc < 1.0:
        print("\nMatriz de Confusión:")
        print(confusion_matrix(y_test, y_pred))
    
    #  Guardar
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print(f"\n Modelo Ensemble guardado en: {model_path}")

if __name__ == "__main__":
    entrenar_ensemble()