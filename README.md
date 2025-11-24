# Automatización Inteligente de Tickets de Soporte

Este proyecto implementa un flujo *end-to-end* para la clasificación automática de tickets de soporte técnico utilizando Procesamiento de Lenguaje Natural y Machine Learning.

El sistema ingesta tickets en crudo, limpia el texto, predice su categoría (TI, RRHH, Finanzas, Soporte), calcula su prioridad mediante reglas de negocio y notifica los resultados a una API externa(mock).

##  Requisitos Técnicos
El proyecto fue desarrollado con Python 3.13 y utiliza las siguientes librerías principales:
* **pandas:** Manipulación de datos estructurados.
* **scikit-learn:** Entrenamiento de modelos (Ensemble Learning) y vectorización.
* **nltk:** Preprocesamiento de texto (Stopwords).
* **requests:** Integración con API REST.

##  Instrucciones de Ejecución

1.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Entrenar el modelo (Opcional):**
    Si se desea re-entrenar el modelo con nuevos datos:
    ```bash
    python src/modelo.py
    ```

3.  **Ejecutar el pipeline principal:**
    Se procesa el dataset, clasifica y envía los resultados:
    ```bash
    python src/procesar_tickets.py tickets.csv
    ```

4.  **Ver resultados:**
    El archivo JSON final se generará en: `output/resultado.json`

##  Decisiones Técnicas

### 1. Estrategia de Modelado (Ensemble)
Dada la limitación del dataset (pocos ejemplos por clase), se optó por un Voting Classifier (Ensemble) que combina:
* Naive Bayes (Multinomial): Base probabilística eficiente para texto.
* SVM (Linear SVC): Para maximizar el margen de separación en alta dimensionalidad.
* Logistic Regression: Para robustez en la estimación.

**Resultado:** Se logró aumentar la precisión (Accuracy) de un 40% inicial a un 95-100% en pruebas, mitigando el riesgo de *underfitting*.

### 2. Procesamiento de Texto 
* Data Augmentation: Se aplicó duplicación sintética del dataset de entrenamiento para estabilizar los pesos del modelo.
* N-Grams (1,2): Se utilizaron unigramas y bigramas para capturar contexto más allá de palabras aisladas.
* Limpieza: Normalización a minúsculas, eliminación de signos y stopwords en español.

### 3. Lógica de Negocio (Prioridad)
La prioridad no se infiere con ML, sino mediante reglas determinísticas basadas en palabras clave críticas (ej. "urgente", "crítico", "nómina" -> Alta), garantizando que los incidentes sensibles siempre tengan máxima atención.

##  Estructura del Repositorio

```text
/ia_ticket_automation
│
├── /src
│   ├── limpieza.py          # Lógica de preprocesamiento
│   ├── modelo.py            # Entrenamiento del Ensemble
│   ├── procesar_tickets.py  # Script orquestador principal
│   ├── *.pkl                # Artefactos del modelo (binarios)
│
├── /output
│   └── resultado.json       # Salida estructurada del proceso
│
├── tickets.csv              # Dataset de entrada
└── requirements.txt         # Dependencias
└── README.md                # Documentación 
```

##  Mejoras Propuestas 

Aunque este MVP cumple con el flujo funcional *end-to-end*, se han identificado las siguientes áreas de mejora para llevar el proyecto a un entorno productivo empresarial de alto nivel:

###  1. Integración con Entornos Reales (API Real)
Actualmente el sistema consume un *mock* (`jsonplaceholder`). Para producción, se propone:
* **Conexión a ITSM:** Reemplazar el endpoint de prueba por la API de herramientas corporativas como *Jira Service Management*, *ServiceNow* o *Zendesk*.
* **Seguridad y Autenticación:** Implementar manejo de credenciales seguro mediante variables de entorno (`.env`) y autenticación robusta (OAuth2 o API Keys rotativas) en lugar de exponerlas en el código.
* **Manejo de Errores y Reintentos:** Incorporar una lógica de *Exponential Backoff*. Si la API de destino falla, el sistema debería esperar y reintentar enviar el ticket automáticamente en lugar de descartarlo.

###  2. MLOps y Evolución del Modelo

* **Modelos Avanzados :** Si el volumen de datos crece significativamente (>10k tickets), se evaluaría migrar de *Naive Bayes/SVM* a modelos de lenguaje pre-entrenados como **BERT** (`distilbert-base-multilingual`) para capturar mejor el contexto semántico complejo.
* **Detección de "Data Drift":** Monitorear si el vocabulario de los tickets cambia con el tiempo (ej. nuevos términos técnicos) para alertar cuándo es necesario un re-entrenamiento.

###  3. Arquitectura e Infraestructura
* **Dockerización:** Crear un `Dockerfile` para empaquetar la aplicación y sus dependencias (Python, NLTK, Scikit-learn), garantizando que funcione igual en cualquier servidor.
* **Base de Datos Persistente:** En lugar de leer/escribir CSVs locales, conectar el script a una base de datos SQL para llevar un histórico confiable de los tickets procesados y sus predicciones.
