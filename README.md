# FutBotMX-Vision-Lab
Convocatoria Copa FutBotMX 2026 Capítulo Visión por Computadora
# **Copa FutBotMX \- Visión por Computadora**

Este repositorio contiene la solución oficial desarrollada para el reto **Copa FutBotMX "Visión por Computadora"** impulsado por la **Secihti** y **Meta**.

El sistema implementa un pipeline de análisis de video de extremo a extremo que procesa partidos de fútbol robótico, extrayendo analíticas avanzadas (posesión, goles, eventos clave, minimapa táctico) mediante la integración de **YOLOv8**, **SAM 3 (Segment Anything Model 3\)** y homografía.

## **Descripción del Enfoque y Arquitectura de la Solución**

Nuestra solución se centra en la eficiencia computacional, la solidez matemática y la innovación analítica. El sistema no solo detecta, sino que comprende el flujo del juego a través de una arquitectura basada en estados.

El pipeline se compone de las siguientes fases:

1. **Fase 0: Creación del Dataset:** Para generar la base de datos de entrenamiento del modelo YOLOv8, se desarrolló un script en Python que extrajo fotogramas clave de los videos de prueba proporcionados, al cual se le aplicó fine tuning. Posteriormente, se utilizó la plataforma **Roboflow** en conjunto con el etiquetado automático de **SAM 3**, lo que permitió generar cajas delimitadoras y polígonos de alta precisión.  
2. **Fase 1: Calibración Geométrica Asistida (GUI):** Interfaz gráfica (tkinter \+ OpenCV) para definir los límites de la cancha y porterías. Se calcula una matriz de homografía (![][image1]) para proyectar la vista de cámara a un Minimapa 2D perfecto con las medidas reglamentarias (182cm x 243cm).  
3. **Fase 2: Asignación de Equipos:** El sistema detecta los robots iniciales y solicita mediante pop-ups la asignación de equipos. Se extraen **Histogramas HSV (Huellas Digitales de Color)**, permitiendo clasificar a los robots durante todo el partido con alta resiliencia a cambios de iluminación.  
4. **Fase 3: Motor Analítico y Pipeline Core IA:**  
   * **Detección y Segmentación:** YOLOv8 detecta robots y balón. Las cajas delimitadoras operan como *prompts* dinámicos para **SAM 3**, extrayendo máscaras precisas.  
   * **Máquina de Estados de Eventos:** El sistema implementa "resolución diferida" para el balón. Discierne inteligentemente entre **Pases, Intercepciones y Tiros (a gol o fallados)** analizando la trayectoria, el tiempo de vuelo y el robot de origen/destino.  
   * **Sensores de Gol:** Las porterías operan con "líneas" basadas en coordenadas ![][image2] absolutas, incluyendo un reset geográfico (el balón debe volver 150px al centro) para evitar parpadeos por *motion blur* o colisiones en la red.  
   * **HUD Táctico:** Visualización en vivo de métricas de posesión, eficiencia de pases y un Minimapa 2D termográfico.

## **Requisitos de Hardware y Software**

**Hardware Recomendado:**

* **CPU:** Mínimo 8 núcleos (Ej. Intel Core i7 / AMD Ryzen 7).  
* **RAM:** 16 GB o superior.  
* **GPU (Crucial):** Tarjeta gráfica NVIDIA compatible con CUDA. Recomendada **NVIDIA RTX 3060, RTX 4070** o superior con al menos 8GB de VRAM para procesar SAM 3 y YOLOv8 simultáneamente a buena velocidad.

**Software y Dependencias:**

* Sistema Operativo: Windows 10/11.  
* Python 3.9 o superior.  
* Drivers: CUDA Toolkit 11.8+ y cuDNN instalados.

**Librerías principales (requirements.txt):**

torch\>=2.0.0  
torchvision\>=0.15.0  
ultralytics==8.1.0  
supervision==0.19.0  
opencv-python\>=4.8.0  
numpy\>=1.24.0  
matplotlib\>=3.7.0

## **Instrucciones de Instalación y Reproducción**

**1\. Clonar el repositorio:**

git clone https://github.com/irvingcal/FutBotMX-Vision-Lab.git

cd FutBotMX-Vision

**2\. Crear entorno virtual (Recomendado con Conda):**

conda create \-n futbot\_env python=3.11  
conda activate futbot\_env

**3\. Instalar dependencias:**

pip install \-r requirements.txt

*(Asegúrate de instalar la versión de PyTorch correspondiente a tu versión de CUDA desde el sitio oficial).*

**4\. Descargar los Modelos:**

Verifica que los siguientes archivos estén en la raíz del proyecto:

* futbot.pt (Modelo YOLOv8 entrenado para detectar robots y balones).  
* sam3.pt (Modelo de SAM 3, deben solicitarlo desde su página web https://huggingface.co/facebook/sam3 ).

**5\. Ejecución:**

**REQUISITO DEL VIDEO (CÁMARA ESTÁTICA):** El video seleccionado para el procesamiento **debe ser grabado con una cámara completamente estática** (sin movimientos de paneo, zoom o inclinación). Esto es vital para que la matriz de homografía inicial se mantenga válida y las analíticas espaciales del mapa táctico sean precisas durante todo el partido.

python procesamiento\_futbot.py

* **Interfaz de Selección:** El sistema abrirá un explorador de archivos para que selecciones el video .mp4.  
* **Modo de Procesamiento:** Un pop-up preguntará si deseas usar el **Modo Turbo** (procesamiento rápido con *frame skipping* para pruebas) o **Calidad Máxima** (renderizado completo).  
* **Calibración:** Sigue las instrucciones en pantalla para marcar las esquinas y porterías. Luego, asigna los IDs a los equipos cuando se te solicite (solo se debe seleccionar un robot por equipo, de preferencia el que se note más).  
* **Aborto Seguro:** Durante el procesamiento, puedes presionar la tecla q en la vista previa para detener el análisis, guardar el video hasta ese punto y exportar las gráficas de forma segura.

## **Resultados Obtenidos**

El sistema exporta automáticamente una carpeta llamada Resultados\_\[NombreVideo\] que contiene el video analizado y una subcarpeta reportes\_analiticos/ con las métricas del encuentro.

### **1\. Head-Up Display (HUD) en Video**

<img width="647" height="992" alt="Detecciones_SAM_YOLO" src="https://github.com/user-attachments/assets/877a57f6-267b-4a94-8e5c-1ebeef6ed4eb" />


*Fig 1\. Interfaz final con segmentación (SAM 3), tracking, panel de eventos en vivo y Minimapa 2D termográfico en la esquina.*

### **2\. Mapas de Tiros (Shot Maps)**

El sistema genera mapas de tiro globales y segmentados por equipo, clasificando los eventos en Goles (Verde), Tiros Fallados (Rojo) e Intercepciones (Azul).

<img width="728" height="972" alt="shot_map_global" src="https://github.com/user-attachments/assets/3411c868-d58e-4e75-a10d-17b28d8ee9c9" />


*Fig 2\. Mapa de Tiros.*

### **3\. Mapas de Calor de Actividad**

Visualización de las zonas de mayor dominio táctico, exportando un mapa global y mapas individuales para el Equipo A y Equipo B.

<img width="728" height="972" alt="mapa_calor_Global" src="https://github.com/user-attachments/assets/77da0899-f013-4980-9d1a-bba78da81a2a" />


*Fig 3\. Mapa de calor de actividad acumulada.*

### **4\. Gráficas de Rendimiento**

Gráficas de barras comparativas sobre la Posesión del Balón y la Eficiencia de Pases.

<img width="600" height="400" alt="grafica_posesion" src="https://github.com/user-attachments/assets/8b30ad17-c93a-4a5d-9acf-051a0edb4ce1" />


*Fig 4\. Análisis de posesión.*

## **Reel de Demostración (Instagram)**

🔗 [**Ver Reel del Proyecto en Instagram**](https://www.instagram.com/reel/DZ0MSRZxC_5/?igsh=MW1xMDZjc3c5d3c1cg==)


## **Licencia y Créditos**

Este proyecto se distribuye bajo la licencia **MIT License** (ver archivo LICENSE en el repositorio).

**Créditos de Tecnologías y Modelos:**

* **Secihti y Federación Mexicana de Robótica:** Por los datasets y el reto Copa FutBotMX.  
* **Meta AI:** Por el modelo fundacional [SAM 3 (Segment Anything Model)](https://github.com/facebookresearch/sam3).  
* **Ultralytics:** Por la arquitectura de detección [YOLOv8](https://github.com/ultralytics/ultralytics).  
* **Roboflow:** Por la librería gráfica [Supervision](https://github.com/roboflow/supervision) utilizada para visualizaciones avanzadas.

