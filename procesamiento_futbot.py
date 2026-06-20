import cv2
import numpy as np
import torch
import os
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from ultralytics import YOLO, SAM
import supervision as sv
import matplotlib.pyplot as plt

# ==========================================
# CONSTANTES Y CONFIGURACIÓN GLOBAL
# ==========================================
puntos_calibracion = []
HISTORIAL_IDS = {}

# Dimensiones Escala 4px = 1cm (Basado en diagrama: 182cm x 243cm)
ANCHO_MAPA = 182 * 4  # 728 px
ALTO_MAPA = 243 * 4   # 972 px

PUNTOS_MAPA_2D = np.array([
    [0, 0],                   # Sup-Izq
    [ANCHO_MAPA, 0],          # Sup-Der
    [ANCHO_MAPA, ALTO_MAPA],  # Inf-Der
    [0, ALTO_MAPA]            # Inf-Izq
], dtype=np.float32)

# ==========================================
# FUNCIONES GRÁFICAS DEL MAPA 2D
# ==========================================
def dibujar_cancha_2d():
    """Dibuja la cancha matemáticamente perfecta basada en el diagrama oficial"""
    img = np.zeros((ALTO_MAPA, ANCHO_MAPA, 3), dtype=np.uint8)
    img[:] = (50, 160, 50) # Fondo verde pasto
    
    # Paredes exteriores (Grosor visual)
    cv2.rectangle(img, (0, 0), (ANCHO_MAPA, ALTO_MAPA), (20, 20, 20), 12)
    
    # Margen interior de 12cm (48 px)
    margen = 48
    cv2.rectangle(img, (margen, margen), (ANCHO_MAPA-margen, ALTO_MAPA-margen), (255, 255, 255), 4)
    
    # Línea central
    centro_y = ALTO_MAPA // 2
    centro_x = ANCHO_MAPA // 2
    cv2.line(img, (margen, centro_y), (ANCHO_MAPA-margen, centro_y), (255, 255, 255), 4)
    
    # Círculo central (60cm diametro -> 120px radio)
    cv2.circle(img, (centro_x, centro_y), 120, (255, 255, 255), 4)
    cv2.circle(img, (centro_x, centro_y), 6, (255, 255, 255), -1)
    
    # Áreas de Portería (80cm ancho x 25cm alto -> 320px x 100px)
    cv2.rectangle(img, (centro_x - 160, margen), (centro_x + 160, margen + 100), (255, 255, 255), 4)
    cv2.rectangle(img, (centro_x - 160, ALTO_MAPA - margen - 100), (centro_x + 160, ALTO_MAPA - margen), (255, 255, 255), 4)
    
    # Porterías Físicas (Amarillo arriba, Azul abajo)
    cv2.rectangle(img, (centro_x - 120, 0), (centro_x + 120, margen), (0, 255, 255), -1) # Amarillo
    cv2.rectangle(img, (centro_x - 120, ALTO_MAPA - margen), (centro_x + 120, ALTO_MAPA), (255, 100, 50), -1) # Azul
    
    return img

def proyectar_punto(bbox, H):
    """Convierte el centro inferior de una caja 2D al plano de la cancha"""
    x1, y1, x2, y2 = bbox
    p = np.array([[[ (x1 + x2) / 2, y2 ]]], dtype=np.float32)
    trans = cv2.perspectiveTransform(p, H)
    return int(trans[0][0][0]), int(trans[0][0][1])

# ==========================================
# INTERFACES DE CALIBRACIÓN (FASES 1 Y 2)
# ==========================================
def dibujar_interfaz_calibracion(frame_base, puntos):
    frame_mostrar = frame_base.copy()
    h, w = frame_mostrar.shape[:2]
    instrucciones = [
        "1. CANCHA EXTERIOR: Sup-Izq", "2. CANCHA EXTERIOR: Sup-Der", 
        "3. CANCHA EXTERIOR: Inf-Der", "4. CANCHA EXTERIOR: Inf-Izq",
        "5. GOL AMARILLO: Poste Izq", "6. GOL AMARILLO: Poste Der",
        "7. GOL AZUL: Poste Izq", "8. GOL AZUL: Poste Der"
    ]
    idx = min(len(puntos), 8)
    
    overlay = frame_mostrar.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    panel_w = 420
    cv2.rectangle(overlay, (w - panel_w, 70), (w, 450), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame_mostrar, 0.3, 0, frame_mostrar)
    
    if idx < 8:
        cv2.putText(frame_mostrar, f"SIGUIENTE: {instrucciones[idx]}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    else:
        cv2.putText(frame_mostrar, "COMPLETADO. Presiona una tecla.", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
    for i, pt in enumerate(puntos):
        cv2.putText(frame_mostrar, f"[{i+1}] {pt[0]}, {pt[1]}", (w - panel_w + 20, 110 + (i*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(frame_mostrar, (pt[0], pt[1]), 8, (0, 0, 255), -1)
        cv2.putText(frame_mostrar, str(i+1), (pt[0] + 15, pt[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    return frame_mostrar

def click_evento_geometria(event, x, y, flags, param):
    global puntos_calibracion
    if event == cv2.EVENT_LBUTTONDOWN and len(puntos_calibracion) < 8:
        puntos_calibracion.append([x, y])
        cv2.imshow("Fase 1: Calibracion", dibujar_interfaz_calibracion(param['frame_limpio'], puntos_calibracion))

def calibrar_geometria(video_path):
    global puntos_calibracion
    puntos_calibracion = []
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    frame_limpio = frame.copy()
    cv2.namedWindow("Fase 1: Calibracion", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fase 1: Calibracion", 1280, 720) # FORZAR VENTANA GRANDE
    cv2.imshow("Fase 1: Calibracion", dibujar_interfaz_calibracion(frame_limpio, puntos_calibracion))
    cv2.setMouseCallback("Fase 1: Calibracion", click_evento_geometria, {'frame_limpio': frame_limpio})
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return np.array(puntos_calibracion[0:4], dtype=np.int32), (tuple(puntos_calibracion[4]), tuple(puntos_calibracion[5])), (tuple(puntos_calibracion[6]), tuple(puntos_calibracion[7]))

def extraer_huella_color(recorte_bgr):
    hsv = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2HSV)
    mask_valida = cv2.bitwise_not(cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255])))
    hist = cv2.calcHist([hsv], [0, 1], mask_valida, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist

def calibrar_equipos(video_path, model_yolo, puntos_cancha):
    cap = cv2.VideoCapture(video_path)
    zona_cancha = sv.PolygonZone(polygon=puntos_cancha)
    frame_valido = None
    recortes_validos = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        det = sv.Detections.from_ultralytics(model_yolo(frame, conf=0.5, verbose=False)[0])
        det = det[zona_cancha.trigger(detections=det)]
        det = det[det.class_id == 1]
        
        if len(det.xyxy) >= 2:
            frame_valido = frame.copy()
            for i, bbox in enumerate(det.xyxy):
                x1, y1, x2, y2 = map(int, bbox)
                recortes_validos.append(frame[max(0,y1):y2, max(0,x1):x2])
                cv2.rectangle(frame_valido, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(frame_valido, f"[{i}]", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            break
            
    cap.release()
    cv2.namedWindow("Fase 2: Equipos", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fase 2: Equipos", 1280, 720) # FORZAR VENTANA GRANDE
    cv2.imshow("Fase 2: Equipos", frame_valido)
    cv2.waitKey(1)
    
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    id_a = simpledialog.askinteger("Equipo A (Superior)", "Ingresa el ID del robot EQUIPO A (inicia arriba):", parent=root, minvalue=0, maxvalue=len(recortes_validos)-1)
    id_b = simpledialog.askinteger("Equipo B (Inferior)", "Ingresa el ID del robot EQUIPO B (inicia abajo):", parent=root, minvalue=0, maxvalue=len(recortes_validos)-1)
    cv2.destroyAllWindows()
    
    if id_a is None: id_a = 0
    if id_b is None: id_b = 1
    
    return extraer_huella_color(recortes_validos[id_a]), extraer_huella_color(recortes_validos[id_b])

# ==========================================
# GENERACIÓN DE REPORTES (MATPLOTLIB)
# ==========================================
def exportar_reportes(stats, calor_A, calor_B, calor_Global, tiros, dir_salida):
    print("\nGenerando Reportes Analíticos Profesionales...")
    os.makedirs(dir_salida, exist_ok=True)
    
    plt.figure(figsize=(6, 4))
    eqs = ['Equipo A', 'Equipo B']
    tot = stats['posesion']['A'] + stats['posesion']['B']
    vals = [stats['posesion']['A']/tot*100 if tot>0 else 0, stats['posesion']['B']/tot*100 if tot>0 else 0]
    plt.bar(eqs, vals, color=['#FF3333', '#33FFFF'])
    plt.title('Porcentaje de Posesión del Balón')
    plt.ylabel('% de Tiempo')
    plt.savefig(os.path.join(dir_salida, 'grafica_posesion.png'))
    plt.close()
    
    plt.figure(figsize=(6, 4))
    pases_A = stats['pases']['A']
    pases_B = stats['pases']['B']
    plt.bar(eqs, [pases_A, pases_B], color=['#FF3333', '#33FFFF'])
    plt.title('Pases Completados por Equipo')
    plt.savefig(os.path.join(dir_salida, 'grafica_pases.png'))
    plt.close()
    
    def crear_shotmap(tiros_filtrados, filename, titulo="Shot Map"):
        img_shot = dibujar_cancha_2d()
        for tiro in tiros_filtrados:
            color = (0, 255, 0) if tiro['res'] == 'Gol' else (0, 0, 255) if tiro['res'] == 'Fallo' else (255, 100, 100)
            cv2.circle(img_shot, (tiro['x'], tiro['y']), 12, color, -1)
            cv2.circle(img_shot, (tiro['x'], tiro['y']), 12, (255,255,255), 2)
        cv2.putText(img_shot, "VERDE: Gol | ROJO: Fallo | AZUL: Interceptado", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(img_shot, titulo, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(dir_salida, filename), img_shot)
        
    crear_shotmap(tiros, 'shot_map_global.png', 'Shot Map: Global')
    crear_shotmap([t for t in tiros if t['eq'] == 'A'], 'shot_map_equipo_A.png', 'Shot Map: Equipo A')
    crear_shotmap([t for t in tiros if t['eq'] == 'B'], 'shot_map_equipo_B.png', 'Shot Map: Equipo B')
    
    for nombre, matriz in [('Global', calor_Global), ('Equipo_A', calor_A), ('Equipo_B', calor_B)]:
        blur = cv2.GaussianBlur(matriz, (51, 51), 0)
        norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        calor_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        base = dibujar_cancha_2d()
        mask = norm > 5
        base[mask] = cv2.addWeighted(base[mask], 0.4, calor_color[mask], 0.6, 0)
        cv2.imwrite(os.path.join(dir_salida, f'mapa_calor_{nombre}.png'), base)

# ==========================================
# FASE 3: PIPELINE DE PROCESAMIENTO
# ==========================================
def procesar_partido(video_path, output_path, puntos_cancha, linea_sup, linea_inf, huella_A, huella_B, modo_turbo=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_yolo = YOLO('futbot.pt').to(device) 
    model_sam = SAM('sam3.pt').to(device)     
    
    tracker_robots = sv.ByteTrack(lost_track_buffer=90)
    tracker_pelota = sv.ByteTrack(lost_track_buffer=15) 
    
    stats = {
        'goles': {'A': 0, 'B': 0},
        'posesion': {'A': 0, 'B': 0},
        'pases': {'A': 0, 'B': 0},
        'intercepciones': {'A': 0, 'B': 0},
        'tiros': {'A': 0, 'B': 0},
        'colisiones': 0
    }
    historial_tiros = []
    
    mapa_calor_A = np.zeros((ALTO_MAPA, ANCHO_MAPA), dtype=np.float32)
    mapa_calor_B = np.zeros((ALTO_MAPA, ANCHO_MAPA), dtype=np.float32)
    mapa_calor_Global = np.zeros((ALTO_MAPA, ANCHO_MAPA), dtype=np.float32)
    
    estado_balon = {
        'equipo_origen': None,
        'id_origen': None,
        'pos_origen': None,
        'en_transito': False,
        'es_tiro_pendiente': False,
        'frames_transito': 0
    }
    
    cooldown_colision = 0
    
    # --- PARÁMETROS ESTRICTOS DE GOL (Ejes Y y X) ---
    Y_GOL_SUP = (linea_sup[0][1] + linea_sup[1][1]) / 2
    Y_GOL_INF = (linea_inf[0][1] + linea_inf[1][1]) / 2
    CENTRO_Y = (Y_GOL_SUP + Y_GOL_INF) / 2
    MARGEN_RESET = abs(Y_GOL_INF - Y_GOL_SUP) * 0.2  
    
    # Límites físicos de los postes (Eje X) con tolerancia de 30 pixeles
    X_MIN_SUP = min(linea_sup[0][0], linea_sup[1][0]) - 30
    X_MAX_SUP = max(linea_sup[0][0], linea_sup[1][0]) + 30
    X_MIN_INF = min(linea_inf[0][0], linea_inf[1][0]) - 30
    X_MAX_INF = max(linea_inf[0][0], linea_inf[1][0]) + 30
    
    estado_gol = "NINGUNO"
    frames_en_centro = 0 # Debounce Anti-Parpadeo
    
    registro_eventos = ["¡Inicia el partido!"]
    
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out_fps = max(1, fps // 3) if modo_turbo else fps
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (width, height))
    
    zona_cancha = sv.PolygonZone(polygon=puntos_cancha)
    mascara_cancha_sam = None 
    H_estatica, _ = cv2.findHomography(puntos_cancha.astype(np.float32), PUNTOS_MAPA_2D)
    
    box_A = sv.BoxAnnotator(color=sv.Color.from_hex("#FF3333"))
    lbl_A = sv.LabelAnnotator(color=sv.Color.from_hex("#FF3333"))
    mask_A = sv.MaskAnnotator(color=sv.Color.from_hex("#FF3333"))
    trc_A = sv.TraceAnnotator(color=sv.Color.from_hex("#FF3333"), thickness=2, trace_length=40)
    
    box_B = sv.BoxAnnotator(color=sv.Color.from_hex("#33FFFF"))
    lbl_B = sv.LabelAnnotator(color=sv.Color.from_hex("#33FFFF"))
    mask_B = sv.MaskAnnotator(color=sv.Color.from_hex("#33FFFF"))
    trc_B = sv.TraceAnnotator(color=sv.Color.from_hex("#33FFFF"), thickness=2, trace_length=40)
    
    box_balon = sv.BoxAnnotator(color=sv.Color.from_hex("#FFFF00"))
    lbl_balon = sv.LabelAnnotator(color=sv.Color.from_hex("#FFFF00"), text_color=sv.Color.from_hex("#000000"))

    def obtener_mascara_segura(res_sam):
        if res_sam.masks is not None:
            mask_cruda = res_sam.masks.data[0].cpu().numpy()
            if mask_cruda.shape != (height, width):
                mask_cruda = cv2.resize(mask_cruda.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            return mask_cruda.astype(bool)
        return np.zeros((height, width), dtype=bool)

    texto_velocidad = "TURBO (Rápido)" if modo_turbo else "MÁXIMA CALIDAD (Lento)"
    print(f"\nIniciando procesamiento [{texto_velocidad}]... (Presiona 'q' en la vista previa para abortar y guardar)")
    
    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            if modo_turbo and frame_count % 3 != 0: continue
            
            if mascara_cancha_sam is None:
                x, y, w, h = cv2.boundingRect(puntos_cancha)
                res = model_sam(frame, bboxes=[[x, y, x+w, y+h]], verbose=False)[0]
                if res.masks is not None:
                    mask_cruda = res.masks.data[0].cpu().numpy()
                    mascara_cancha_sam = cv2.resize(mask_cruda.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)

            res_yolo = model_yolo(frame, conf=0.4, verbose=False)[0]
            det = sv.Detections.from_ultralytics(res_yolo).with_nms(threshold=0.5, class_agnostic=False)
            
            pelotas = det[det.class_id == 0]
            if len(pelotas) > 1: pelotas = pelotas[np.argmax(pelotas.confidence):np.argmax(pelotas.confidence)+1]
            tracked_pelota = tracker_pelota.update_with_detections(pelotas)
            
            det_robots = det[zona_cancha.trigger(detections=det)]
            tracked_robots = tracker_robots.update_with_detections(det_robots[det_robots.class_id == 1])
            
            pos_balon_2d = proyectar_punto(pelotas.xyxy[0], H_estatica) if len(pelotas) > 0 else None
            
            id_toca = None
            eq_toca = None
            dist_min = float('inf')
            posiciones_robots = []
            mascaras_robots = []
            
            for i in range(len(tracked_robots.xyxy)):
                bbox = tracked_robots.xyxy[i]
                tid = tracked_robots.tracker_id[i]
                
                if tid not in HISTORIAL_IDS:
                    x1, y1, x2, y2 = map(int, bbox)
                    recorte = frame[max(0,y1):y2, max(0,x1):x2]
                    if recorte.size > 0:
                        sim_A = cv2.compareHist(extraer_huella_color(recorte), huella_A, cv2.HISTCMP_CORREL)
                        sim_B = cv2.compareHist(extraer_huella_color(recorte), huella_B, cv2.HISTCMP_CORREL)
                        HISTORIAL_IDS[tid] = "A" if sim_A > sim_B else "B"
                
                eq = HISTORIAL_IDS[tid]
                pos_r = proyectar_punto(bbox, H_estatica)
                posiciones_robots.append(pos_r)
                
                cv2.circle(mapa_calor_Global, pos_r, 15, 1.0, -1)
                if eq == "A": cv2.circle(mapa_calor_A, pos_r, 15, 1.0, -1)
                else: cv2.circle(mapa_calor_B, pos_r, 15, 1.0, -1)
                
                if pos_balon_2d:
                    dist = np.linalg.norm(np.array(pos_r) - np.array(pos_balon_2d))
                    if dist < 80 and dist < dist_min:
                        dist_min, id_toca, eq_toca = dist, tid, eq
                        
                res_sam_robot = model_sam(frame, bboxes=[bbox], verbose=False)[0]
                mascaras_robots.append(obtener_mascara_segura(res_sam_robot))
            
            # --- LÓGICA DE EVENTOS ---
            if eq_toca:
                stats['posesion'][eq_toca] += 1
                
                if estado_balon['en_transito']:
                    if eq_toca == estado_balon['equipo_origen']:
                        if id_toca != estado_balon['id_origen']:
                            stats['pases'][eq_toca] += 1
                            registro_eventos.append(f"> Pase completado: {eq_toca}")
                    else:
                        stats['intercepciones'][eq_toca] += 1
                        registro_eventos.append(f"> ¡Robo/Intercepción de {eq_toca}!")
                        
                    estado_balon['en_transito'] = False
                    estado_balon['es_tiro_pendiente'] = False
                    
                estado_balon['equipo_origen'] = eq_toca
                estado_balon['id_origen'] = id_toca
                estado_balon['pos_origen'] = pos_balon_2d
                estado_balon['frames_transito'] = 0
                
            elif estado_balon['equipo_origen'] and pos_balon_2d:
                if not estado_balon['en_transito']:
                    estado_balon['en_transito'] = True
                    estado_balon['frames_transito'] = 0
                else:
                    estado_balon['frames_transito'] += 1
                    
                    if estado_balon['frames_transito'] > 45: # Viajó por más de 1.5s solo
                        delta_y = pos_balon_2d[1] - estado_balon['pos_origen'][1]
                        
                        es_tiro_fallado = (estado_balon['equipo_origen'] == "A" and delta_y > 150) or \
                                          (estado_balon['equipo_origen'] == "B" and delta_y < -150)
                        
                        if es_tiro_fallado:
                            stats['tiros'][estado_balon['equipo_origen']] += 1
                            historial_tiros.append({
                                'x': estado_balon['pos_origen'][0], 
                                'y': estado_balon['pos_origen'][1], 
                                'eq': estado_balon['equipo_origen'], 
                                'res': 'Fallo'
                            })
                            registro_eventos.append(f"> Tiro fallado de {estado_balon['equipo_origen']}")
                        
                        estado_balon['en_transito'] = False
                        estado_balon['equipo_origen'] = None

            # --- LÓGICA DE SENSORES DE GOL (ESTRICTOS EN X, Y, y DEBOUNCE) ---
            if len(pelotas) > 0:
                cx_balon = (pelotas.xyxy[0][0] + pelotas.xyxy[0][2]) / 2
                cy_balon = (pelotas.xyxy[0][1] + pelotas.xyxy[0][3]) / 2
                
                # Check para Reset de Gol (Debounce por Fotogramas)
                if (CENTRO_Y - MARGEN_RESET) < cy_balon < (CENTRO_Y + MARGEN_RESET):
                    frames_en_centro += 1
                    if frames_en_centro > 10: # Debe estar 10 frames en el centro genuinamente
                        estado_gol = "NINGUNO"
                else:
                    frames_en_centro = 0
                
                # Evaluación de Goles
                if cy_balon < Y_GOL_SUP and (X_MIN_SUP <= cx_balon <= X_MAX_SUP) and estado_gol != "SUP":
                    stats['goles']['B'] += 1
                    estado_gol = "SUP"
                    registro_eventos.append(f"> ¡GOL EQUIPO B! ({stats['goles']['A']} - {stats['goles']['B']})")
                    
                    if estado_balon['equipo_origen'] == 'B':
                        if not estado_balon['es_tiro_pendiente']:
                            # tiro lejano
                            stats['tiros']['B'] += 1
                            p_2d = estado_balon['pos_origen'] if estado_balon['pos_origen'] else pos_balon_2d
                            if p_2d: historial_tiros.append({'x': p_2d[0], 'y': p_2d[1], 'eq': 'B', 'res': 'Gol'})
                        else:
                            
                            if historial_tiros: historial_tiros[-1]['res'] = 'Gol'
                            
                    estado_balon['en_transito'] = False
                    estado_balon['equipo_origen'] = None
                        
                elif cy_balon > Y_GOL_INF and (X_MIN_INF <= cx_balon <= X_MAX_INF) and estado_gol != "INF":
                    stats['goles']['A'] += 1
                    estado_gol = "INF"
                    registro_eventos.append(f"> ¡GOL EQUIPO A! ({stats['goles']['A']} - {stats['goles']['B']})")
                    
                    if estado_balon['equipo_origen'] == 'A':
                        if not estado_balon['es_tiro_pendiente']:
                            stats['tiros']['A'] += 1
                            p_2d = estado_balon['pos_origen'] if estado_balon['pos_origen'] else pos_balon_2d
                            if p_2d: historial_tiros.append({'x': p_2d[0], 'y': p_2d[1], 'eq': 'A', 'res': 'Gol'})
                        else:
                            if historial_tiros: historial_tiros[-1]['res'] = 'Gol'
                            
                    estado_balon['en_transito'] = False
                    estado_balon['equipo_origen'] = None

            if cooldown_colision == 0 and len(posiciones_robots) > 1:
                for i in range(len(posiciones_robots)):
                    for j in range(i+1, len(posiciones_robots)):
                        if np.linalg.norm(np.array(posiciones_robots[i]) - np.array(posiciones_robots[j])) < 50:
                            stats['colisiones'] += 1
                            cooldown_colision = 30
            if cooldown_colision > 0: cooldown_colision -= 1

            # --- RENDERIZADO VISUAL ---
            frame_anotado = frame.copy()
            if mascara_cancha_sam is not None:
                capa_verde = np.zeros_like(frame_anotado); capa_verde[:] = (0, 100, 0)
                frame_anotado = np.where(mascara_cancha_sam[:, :, None], cv2.addWeighted(frame_anotado, 0.8, capa_verde, 0.2, 0), frame_anotado)
            
            cv2.line(frame_anotado, linea_sup[0], linea_sup[1], (255, 0, 255), 3)
            cv2.line(frame_anotado, linea_inf[0], linea_inf[1], (255, 255, 0), 3)
            
            if len(pelotas) > 0:
                frame_anotado = box_balon.annotate(scene=frame_anotado, detections=pelotas)
                frame_anotado = lbl_balon.annotate(scene=frame_anotado, detections=pelotas, labels=["Balon"])
            
            if len(tracked_robots) > 0:
                tracked_robots.mask = np.array(mascaras_robots)
                idx_A = [i for i, tid in enumerate(tracked_robots.tracker_id) if HISTORIAL_IDS.get(tid) == "A"]
                idx_B = [i for i, tid in enumerate(tracked_robots.tracker_id) if HISTORIAL_IDS.get(tid) == "B"]
                
                if idx_A:
                    det_A = tracked_robots[idx_A]
                    frame_anotado = mask_A.annotate(scene=frame_anotado, detections=det_A)
                    frame_anotado = trc_A.annotate(scene=frame_anotado, detections=det_A)
                    frame_anotado = box_A.annotate(scene=frame_anotado, detections=det_A)
                    frame_anotado = lbl_A.annotate(scene=frame_anotado, detections=det_A, labels=[f"A-{t}" for t in det_A.tracker_id])
                if idx_B:
                    det_B = tracked_robots[idx_B]
                    frame_anotado = mask_B.annotate(scene=frame_anotado, detections=det_B)
                    frame_anotado = trc_B.annotate(scene=frame_anotado, detections=det_B)
                    frame_anotado = box_B.annotate(scene=frame_anotado, detections=det_B)
                    frame_anotado = lbl_B.annotate(scene=frame_anotado, detections=det_B, labels=[f"B-{t}" for t in det_B.tracker_id])
                
            # --- TABLEROS HUD TÁCTICOS ---
            cv2.rectangle(frame_anotado, (20, 20), (350, 100), (0,0,0), -1)
            cv2.putText(frame_anotado, f"GOL equipo A: {stats['goles']['A']}", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
            cv2.putText(frame_anotado, f"GOL equipo B: {stats['goles']['B']}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
            
            tot_pos = stats['posesion']['A'] + stats['posesion']['B']
            pct_A = int((stats['posesion']['A'] / tot_pos)*100) if tot_pos>0 else 0
            pct_B = 100 - pct_A if tot_pos>0 else 0
            
            tot_pases = stats['pases']['A'] + stats['pases']['B']
            pct_pas_A = int((stats['pases']['A'] / tot_pases)*100) if tot_pases>0 else 0
            pct_pas_B = 100 - pct_pas_A if tot_pases>0 else 0
            
            cv2.rectangle(frame_anotado, (20, 120), (350, 320), (0,0,0), -1)
            cv2.putText(frame_anotado, f"Posesion: A({pct_A}%) B({pct_B}%)", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame_anotado, f"Pases: A({pct_pas_A}%) B({pct_pas_B}%)", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame_anotado, f"Tiros a gol: A({stats['tiros']['A']}) B({stats['tiros']['B']})", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame_anotado, f"Robos: A({stats['intercepciones']['A']}) B({stats['intercepciones']['B']})", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame_anotado, f"Colisiones Totales: {stats['colisiones']}", (30, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,150,255), 2)
            
            cv2.rectangle(frame_anotado, (20, 340), (450, 480), (0, 0, 0), -1)
            cv2.putText(frame_anotado, "EVENTOS DEL PARTIDO:", (30, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for idx, evento in enumerate(registro_eventos[-4:]):
                cv2.putText(frame_anotado, f"{evento}", (30, 400 + (idx * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            minimapa_base = dibujar_cancha_2d()
            if pos_balon_2d: cv2.circle(minimapa_base, pos_balon_2d, 18, (0, 255, 255), -1)
            for i, pos_r in enumerate(posiciones_robots):
                eq = HISTORIAL_IDS.get(tracked_robots.tracker_id[i])
                cv2.circle(minimapa_base, pos_r, 25, (51, 51, 255) if eq == "A" else (255, 255, 51), -1)
                cv2.circle(minimapa_base, pos_r, 25, (255, 255, 255), 3)
                
            mini_resized = cv2.resize(minimapa_base, (int(ANCHO_MAPA*0.3), int(ALTO_MAPA*0.3)))
            h_m, w_m = mini_resized.shape[:2]
            frame_anotado[20:20+h_m, width-w_m-20:width-20] = mini_resized
                
            out.write(frame_anotado)
            
            vista_previa = cv2.resize(frame_anotado, (1280, 720))
            cv2.imshow("Procesando Partido (Presiona 'q' para guardar progreso y salir)", vista_previa)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] Aborto manual detectado. Deteniendo procesamiento y empaquetando video...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupción por teclado (Ctrl+C). Empaquetando video...")
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un fallo en el procesamiento matemático: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("\n" + "="*50)
        print(f"[ÉXITO] VIDEO GUARDADO HASTA EL ÚLTIMO FOTOGRAMA PROCESADO.")
        print(f"Ruta del video: {os.path.abspath(output_path)}")
        print("="*50)
        
        dir_reportes = os.path.join(os.path.dirname(output_path), "reportes_analiticos")
        exportar_reportes(stats, mapa_calor_A, mapa_calor_B, mapa_calor_Global, historial_tiros, dir_reportes)
        print(f"\n[INFO] Gráficas y Mapas de Calor exportados de forma segura en: {os.path.abspath(dir_reportes)}")

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    video_entrada = filedialog.askopenfilename(title="Selecciona el video", filetypes=[("Archivos de video", "*.mp4 *.avi *.mov")])
    if not video_entrada: exit()
    
    modo_turbo = messagebox.askyesno(
        "Configuración de Procesamiento",
        "¿Deseas activar el 'Modo Turbo'?\n\n"
        "SÍ: Procesa el video ~3 veces más rápido saltando fotogramas (Ideal para pruebas rápidas).\n"
        "NO: Procesa a máxima calidad todos los fotogramas (Ideal para el renderizado final)."
    )
        
    dir_name = os.path.dirname(video_entrada)
    base_name = os.path.basename(video_entrada)
    nombre_sin_ext = os.path.splitext(base_name)[0]
    
    carpeta_resultados = os.path.join(dir_name, f"Resultados_{nombre_sin_ext}")
    os.makedirs(carpeta_resultados, exist_ok=True)
    
    video_salida = os.path.join(carpeta_resultados, f"analizado_{base_name}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_yolo_calib = YOLO('futbot.pt').to(device)
    
    try:
        pts_cancha, lin_sup, lin_inf = calibrar_geometria(video_entrada)
        huella_a, huella_b = calibrar_equipos(video_entrada, model_yolo_calib, pts_cancha)
        procesar_partido(video_entrada, video_salida, pts_cancha, lin_sup, lin_inf, huella_a, huella_b, modo_turbo=modo_turbo)
    except Exception as e:
        import traceback
        print(f"Error Crítico: {e}")
        traceback.print_exc()