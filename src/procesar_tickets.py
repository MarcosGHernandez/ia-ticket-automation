import pandas as pd
import pickle
import json
import requests
import sys
import os
import datetime
import re
import nltk
from nltk.corpus import stopwords

# Configuración de seguridad para NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# 1. FUNCIONES DE NEGOCIO 

def limpiar_texto(texto):
    
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    stop_words = set(stopwords.words('spanish'))
    palabras = texto.split()
    return " ".join([p for p in palabras if p not in stop_words])

def calcular_prioridad(texto, categoria):
    """
    Reglas de prioridad .
    Define reglas basadas en palabras clave de urgencia.
    """
    texto = str(texto).lower()
    
    # Palabras que denotan urgencia 
    triggers_alta = ['urgente', 'crítico', 'falla', 'error', 'seguridad', 
                     'cayó', 'servidor', 'robo', 'bloqueo', 'azul']
    
    # Palabras que denotan mantenimiento o peticiones estándar
    triggers_media = ['acceso', 'lento', 'actualizar', 'instalar', 'licencia', 
                      'vacaciones', 'factura', 'reembolso']
    
    if any(t in texto for t in triggers_alta):
        return "Alta"
    elif any(t in texto for t in triggers_media):
        return "Media"
    else:
        return "Baja"

# 2. LÓGICA PRINCIPAL 

def main():
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Manejo de argumentos 
    if len(sys.argv) < 2:
        print("  Advertencia: No indicaste archivo. Usando por defecto 'tickets.csv'")
        archivo_entrada = 'tickets.csv'
    else:
        archivo_entrada = sys.argv[1]

    ruta_csv = os.path.join(base_dir, archivo_entrada)
    model_path = os.path.join(base_dir, 'src', 'modelo_entrenado.pkl')
    vec_path = os.path.join(base_dir, 'src', 'vectorizador.pkl')
    output_json = os.path.join(base_dir, 'output', 'resultado.json')

    print(f" Iniciando procesamiento sobre: {archivo_entrada}")

    # 1. Cargar Modelos
    try:
        with open(model_path, 'rb') as f:
            modelo = pickle.load(f)
        with open(vec_path, 'rb') as f:
            vectorizador = pickle.load(f)
    except FileNotFoundError:
        print(" Error: Falta el modelo (.pkl). Ejecuta primero src/modelo.py")
        return

    # Cargar Datos
    try:
        df = pd.read_csv(ruta_csv)
    except FileNotFoundError:
        print(f" Error: No se encuentra el archivo {ruta_csv}")
        return

    # Normalización de columnas
    df.columns = [c.strip().lower() for c in df.columns]
    # Buscar columna de descripción 
    col_desc = next((c for c in df.columns if 'descrip' in c), None)
    if not col_desc:
        col_desc = df.columns[1] 

    
    print(" Clasificando tickets con IA...")
    df['texto_limpio'] = df[col_desc].apply(limpiar_texto)
    X_vec = vectorizador.transform(df['texto_limpio'])
    df['categoria_predicha'] = modelo.predict(X_vec)

    #  Generar Salida y API
    resultados = []
    url_api = "https://jsonplaceholder.typicode.com/posts" 

    print(f" Enviando peticiones a API Mock ({url_api})...")
    
    for index, row in df.iterrows():
        # Obtener ID 
        t_id = str(row['id']) if 'id' in row else str(index + 1)
        
        cat = row['categoria_predicha']
        texto = row[col_desc]
        prio = calcular_prioridad(texto, cat)
        
        # Construccion objeto JSON final 
        ticket_obj = {
            "ticket_id": t_id,
            "categoria_predicha": cat,
            "prioridad": prio,
            "procesado_en": datetime.datetime.now().isoformat()
        }
        resultados.append(ticket_obj)

        # Enviar a API
        try:
            payload = {"id": t_id, "categoria": cat, "prioridad": prio}
            requests.post(url_api, json=payload, timeout=2)
            print("✔", end="", flush=True) # Indicador visual de éxito
        except:
            print("x", end="", flush=True)

    print("\n Procesamiento API finalizado.")

    # 5. Guardar JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)

    print(f" JSON generado exitosamente en: {output_json}")
    print(" FIN DEL PROCESO")

if __name__ == "__main__":
    main()