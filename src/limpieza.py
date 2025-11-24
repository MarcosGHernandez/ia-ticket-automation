import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords

# Descargas silenciosas de NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def limpiar_texto(texto):

    if not isinstance(texto, str):
        return ""
    
    #  Minúsculas
    texto = texto.lower()
    
    #  Eliminar puntuación (caracteres especiales)
    texto = re.sub(r'[^\w\s]', '', texto)
    
    #  Eliminar stopwords
    stop_words = set(stopwords.words('spanish'))
    palabras = texto.split()
    palabras_limpias = [p for p in palabras if p not in stop_words]
    
    return " ".join(palabras_limpias)

def generar_etiqueta_automatica(texto):
 
    texto = str(texto).lower()
    
    if any(x in texto for x in ['computadora', 'teclado', 'monitor', 'vpn', 'wifi', 'acceso', 'contraseña', 'error', 'sistema', 'laptop', 'servidor']):
        return 'TI'
    elif any(x in texto for x in ['factura', 'impuestos', 'viáticos', 'reembolso', 'pago', 'proveedor', 'banco', 'fiscal', 'costo']):
        return 'Finanzas'
    elif any(x in texto for x in ['vacaciones', 'sueldo', 'nómina', 'seguro', 'contrato', 'beneficios', 'horario', 'baja', 'alta', 'laboral']):
        return 'RRHH'
    else:
        return 'Soporte General'

def procesar():
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Carpeta raíz
    input_file = os.path.join(base_dir, 'tickets.csv')
    output_file = os.path.join(base_dir, 'tickets_procesado.csv')

    print(f" Iniciando limpieza desde: {input_file}")

    try:
       
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("❌ ERROR: No se encontró 'tickets.csv'. Asegúrate de exportar el Excel a CSV en la carpeta raíz.")
        return

    # Normalizar nombres de columnas
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"📋 Columnas detectadas: {list(df.columns)}")

    # Detectar columna de descripción
    col_desc = next((c for c in df.columns if 'descrip' in c), None)
    if not col_desc:
        
        if len(df.columns) > 2:
            col_desc = df.columns[2] 
        else:
             col_desc = df.columns[1]
        print(f"Nombre 'descripcion' no hallado exacto. Usando columna: '{col_desc}'")

    # APLICAR LÓGICA
  
    print("  Etiquetando datos automáticamente...")
    df['categoria'] = df[col_desc].apply(generar_etiqueta_automatica)
    
    #  Limpiar Texto 
    print("dt Limpiando descripciones...")
    df['descripcion_limpia'] = df[col_desc].apply(limpiar_texto)
    
    #  GUARDAR 
  
    cols_finales = ['id'] if 'id' in df.columns else []
    cols_finales += [col_desc, 'descripcion_limpia', 'categoria']
    
    df_final = df[cols_finales]
    df_final.to_csv(output_file, index=False)
    
    print("\n PROCESO COMPLETADO EXITOSAMENTE")
    print("-----------------------------------")
    print(df_final['categoria'].value_counts())
    print(f"\n Archivo generado: {output_file}")

if __name__ == "__main__":
    procesar()