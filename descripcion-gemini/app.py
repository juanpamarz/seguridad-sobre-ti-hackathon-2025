import os
import cv2
import time
import threading
import queue
import numpy as np
import google.generativeai as genai
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuraciones
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Crear directorio de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configurar API key para Gemini
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Modelo de Gemini
model = genai.GenerativeModel('gemini-1.5-flash')

# Variables globales
video_capture = None
is_processing = False
frame_buffer = queue.Queue(maxsize=30)  # Búfer para almacenar frames
current_frame = None  # Frame actual para análisis y visualización
frame_lock = threading.Lock()  # Para acceso sincronizado al frame actual
last_description = "Esperando análisis..."
analysis_interval = 5  # segundos entre análisis
current_video_path = None
processing_thread = None
frame_reader_thread = None

alert_history = []  # Lista para almacenar todas las alertas detectadas
alert_timestamps = []  # Timestamps correspondientes a cada alerta
alert_lock = threading.Lock()  # Para acceso sincronizado a las alertas

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_frame(frame):
    """Analiza un frame con Gemini y retorna la descripción solo si detecta actividad sospechosa"""
    try:
        # Convertir frame a formato compatible con Gemini
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        
        # Preparar la imagen para Gemini
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_bytes
        }
        
        # Prompt enfocado en detectar solo actividades sospechosas
        prompt = """
        Analiza esta imagen de vigilancia y determina si hay actividad sospechosa o ilegal.
        
        IMPORTANTE: Si no hay actividad sospechosa, responde exactamente "NORMAL: No se detecta actividad sospechosa."
        
        Si detectas alguna actividad sospechosa, robo, vandalismo u otra actividad ilegal, describe 
        detalladamente lo que está sucediendo con el siguiente formato:
        "ALERTA: [Descripción detallada de la actividad sospechosa, incluyendo ubicación, personas 
        involucradas, acciones específicas y objetos relevantes]"
        
        Sé muy preciso y solo reporta como sospechoso lo que claramente lo sea.
        """
        
        print("Enviando frame para análisis de actividad sospechosa...")
        
        # Enviar a Gemini para análisis (formato correcto)
        response = model.generate_content(
            contents=[{"role": "user", "parts": [
                {"text": prompt},
                image_part
            ]}]
        )
        
        # Verificar si hay respuesta válida
        if not response:
            print("Error: Respuesta vacía de Gemini")
            return "Error: No se pudo generar análisis"
            
        text = response.text
        
        # Solo actualizar la descripción si se detecta actividad sospechosa
        if text.startswith("ALERTA:"):
            print(f"¡ACTIVIDAD SOSPECHOSA DETECTADA! {text[:100]}...")
            return text
        elif text.startswith("NORMAL:"):
            print("Actividad normal detectada, no se actualiza descripción")
            return None
        else:
            print(f"Respuesta inesperada del modelo: {text[:100]}...")
            return f"Error: Respuesta inesperada - {text[:100]}"
        
    except Exception as e:
        error_msg = f"Error en análisis: {str(e)}"
        print(error_msg)
        return error_msg

def process_video():
    """Procesa el video en un hilo separado, alertando solo de actividades sospechosas"""
    global is_processing, last_description, current_frame, frame_lock, alert_history, alert_timestamps
    
    last_analysis_time = 0
    frame_count = 0
    prev_frame = None
    change_threshold = 30000
    video_start_time = time.time()
    
    print("Iniciando vigilancia de actividad sospechosa...")
    last_description = "Sistema de vigilancia activo. Monitoreando actividad sospechosa..."
    
    # Limpiar historial de alertas al iniciar nuevo procesamiento
    with alert_lock:
        alert_history = []
        alert_timestamps = []
    
    while is_processing:
        # Obtener frame actual para análisis
        with frame_lock:
            if current_frame is None:
                time.sleep(0.1)
                continue
            analysis_frame = current_frame.copy()
            
        frame_count += 1
        
        if frame_count % 150 == 0:
            print(f"Vigilancia activa: frame {frame_count}")
        
        # Detectar cambios y análisis
        should_analyze = False
        current_time = time.time()
        time_based_analysis = (current_time - last_analysis_time > analysis_interval)
        
        if prev_frame is not None and time_based_analysis:
            current_gray = cv2.cvtColor(cv2.resize(analysis_frame, (320, 240)), cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(cv2.resize(prev_frame, (320, 240)), cv2.COLOR_BGR2GRAY)
            
            diff = cv2.absdiff(current_gray, prev_gray)
            non_zero_count = np.count_nonzero(diff)
            
            if non_zero_count > change_threshold:
                should_analyze = True
        
        if frame_count % 10 == 0:
            prev_frame = analysis_frame.copy()
        
        if (time_based_analysis and should_analyze) or frame_count == 30:
            print(f"Analizando actividad en frame {frame_count}")
            analysis_result = analyze_frame(analysis_frame)
            
            # Si se detectó actividad sospechosa, registrarla
            if analysis_result is not None and analysis_result.startswith("ALERTA:"):
                # Calcular tiempo relativo desde el inicio del video
                elapsed_time = current_time - video_start_time
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                timestamp = f"{minutes:02d}:{seconds:02d}"
                
                # Para la interfaz de usuario, solo mostrar la alerta sin timestamp
                last_description = analysis_result
                
                # Para el historial y reporte, guardar la versión con timestamp
                timestamped_alert = f"[{timestamp}] {analysis_result}"
                
                # Agregar al historial de alertas
                with alert_lock:
                    alert_history.append(timestamped_alert)
                    alert_timestamps.append(elapsed_time)
                
                print(f"¡Alerta #{len(alert_history)} generada!: {analysis_result[:100]}...")
                
            last_analysis_time = current_time
            
        time.sleep(0.1)  # Reducir carga de CPU
    
    # Generar informe al finalizar
    generate_police_report()
    print("Vigilancia de video detenida")

def read_frames():
    """Lee frames del video y los almacena en el búfer compartido"""
    global is_processing, current_video_path, frame_buffer, current_frame, frame_lock
    
    local_capture = None
    frame_count = 0
    
    try:
        # Abrir video
        local_capture = cv2.VideoCapture(current_video_path)
        if not local_capture.isOpened():
            print("Error: No se pudo abrir el video en el lector de frames")
            is_processing = False
            return
            
        print(f"Iniciando lectura de frames desde: {current_video_path}")
        
        # Obtener la velocidad original del video (FPS)
        fps = local_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Valor por defecto si no se puede detectar
        frame_delay = 1.0 / fps  # Tiempo entre frames
        
        print(f"Velocidad de reproducción: {fps} FPS (delay: {frame_delay:.4f}s)")
        
        while is_processing:
            success, frame = local_capture.read()
            
            if not success:
                print("Fin del video alcanzado en lector, reiniciando...")
                local_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            frame_count += 1
            
            # Guardar el frame actual para análisis y visualización
            with frame_lock:
                current_frame = frame.copy()
                
            # Mostrar mensaje de depuración cada 100 frames
            if frame_count % 100 == 0:
                print(f"Frames leídos: {frame_count}")
                
            # Controlar velocidad según el FPS original del video
            time.sleep(frame_delay)
            
    except Exception as e:
        print(f"Error en lector de frames: {str(e)}")
    finally:
        print("Finalizando lector de frames")
        if local_capture is not None:
            local_capture.release()

def generate_frames():
    """Genera frames del video para streaming con alertas destacadas"""
    global is_processing, last_description, current_frame, frame_lock
    
    # Variable para almacenar el FPS para la visualización
    fps = 30  # Valor predeterminado
    frame_delay = 1.0 / fps
    
    while True:
        # Si no hay procesamiento activo, mostrar pantalla de espera
        if not is_processing or current_frame is None:
            # Generar un frame en blanco con mensaje
            blank_frame = 255 * np.ones((480, 640, 3), np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(blank_frame, 
                       "Sistema de vigilancia inactivo", 
                       (100, 240), font, 0.8, (0, 0, 0), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.5)
            continue
        
        try:
            # Obtener el frame actual
            with frame_lock:
                display_frame = current_frame.copy()
                
            # Determinar el formato de visualización según si es una alerta o monitoreo normal
            is_alert = "ALERTA:" in last_description
            
            # Agregar recuadro rojo si es una alerta
            if is_alert:
                # Dibujar borde rojo para alertas
                height, width = display_frame.shape[:2]
                cv2.rectangle(display_frame, (0, 0), (width-1, height-1), (0, 0, 255), 10)
                
                # Banner de alerta en la parte superior
                cv2.rectangle(display_frame, (0, 0), (width, 60), (0, 0, 200), -1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(display_frame, "ALERTA DE ACTIVIDAD SOSPECHOSA", 
                           (width//2 - 250, 40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Codificar frame para streaming
            ret, buffer = cv2.imencode('.jpg', display_frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            # Controlar velocidad de reproducción según el FPS del video original
            time.sleep(frame_delay)
            
        except Exception as e:
            print(f"Error en generate_frames: {str(e)}")
            time.sleep(0.1)

def generate_police_report():
    """Genera un informe policial basado en las alertas detectadas"""
    global alert_history, alert_timestamps, current_video_path
    
    if not alert_history:
        print("No se generó informe policial: no se detectaron alertas")
        return None
    
    try:
        # Crear directorio para informes si no existe
        reports_dir = 'reports'
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generar nombre de archivo basado en la fecha y hora
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_name = os.path.basename(current_video_path) if current_video_path else "video_desconocido"
        report_filename = f"{reports_dir}/informe_policial_{timestamp}_{video_name}.txt"
        
        # Obtener fecha y hora actual
        current_datetime = time.strftime("%d/%m/%Y %H:%M:%S")
        
        with open(report_filename, 'w', encoding='utf-8') as report_file:
            report_file.write("=================================================================\n")
            report_file.write(f"INFORME POLICIAL DE VIGILANCIA - GENERADO: {current_datetime}\n")
            report_file.write("=================================================================\n\n")
            report_file.write(f"ARCHIVO DE VIDEO: {video_name}\n")
            report_file.write(f"TOTAL DE INCIDENTES DETECTADOS: {len(alert_history)}\n\n")
            
            report_file.write("CRONOLOGÍA DE INCIDENTES DETECTADOS:\n")
            report_file.write("-----------------------------------------------------------------\n\n")
            
            for i, (alert, timestamp) in enumerate(zip(alert_history, alert_timestamps)):
                report_file.write(f"INCIDENTE #{i+1} - {alert}\n\n")
            
            report_file.write("-----------------------------------------------------------------\n")
            report_file.write("ANÁLISIS FINAL:\n\n")
            
            # Generar un resumen basado en la cantidad y tipo de alertas
            if len(alert_history) >= 5:
                report_file.write("NIVEL DE ALERTA: ALTO - Múltiples incidentes detectados\n")
            elif len(alert_history) >= 2:
                report_file.write("NIVEL DE ALERTA: MEDIO - Varios incidentes detectados\n")
            else:
                report_file.write("NIVEL DE ALERTA: BAJO - Incidente aislado detectado\n")
            
            report_file.write("\nSe recomienda revisión humana para confirmación de incidentes.")
            report_file.write("\n\n=== FIN DEL INFORME ===\n")
        
        print(f"Informe policial generado exitosamente: {report_filename}")
        return report_filename
        
    except Exception as e:
        print(f"Error al generar informe policial: {str(e)}")
        return None

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html', is_processing=is_processing)

@app.route('/video_feed')
def video_feed():
    """Endpoint para el streaming de video"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Maneja la subida de un archivo de video"""
    global current_video_path
    
    if 'video' not in request.files:
        return jsonify({"error": "No se seleccionó ningún archivo"}), 400
        
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({"error": "No se seleccionó ningún archivo"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        current_video_path = filepath
        return jsonify({"success": True, "filename": filename})
    else:
        return jsonify({"error": "Tipo de archivo no permitido"}), 400

@app.route('/start_video', methods=['POST'])
def start_video():
    """Inicia el procesamiento del video"""
    global video_capture, is_processing, current_video_path, processing_thread, frame_reader_thread
    
    if current_video_path is None:
        return jsonify({"error": "No hay video seleccionado"}), 400
    
    # Detener cualquier procesamiento anterior
    stop_video()
    
    try:
        # Abrir el archivo de video
        video_capture = cv2.VideoCapture(current_video_path)
        if not video_capture.isOpened():
            return jsonify({"error": "No se pudo abrir el video"}), 500
            
        # Configurar buffer de video más pequeño para reducir latencia
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            
        # Iniciar el procesamiento
        is_processing = True
        frame_reader_thread = threading.Thread(target=read_frames)
        frame_reader_thread.daemon = True
        frame_reader_thread.start()
        
        processing_thread = threading.Thread(target=process_video)
        processing_thread.daemon = True
        processing_thread.start()
        
        print("Procesamiento de video iniciado correctamente")
        return jsonify({"status": "success"})
    except Exception as e:
        is_processing = False
        if video_capture is not None:
            video_capture.release()
            video_capture = None
        return jsonify({"error": str(e)}), 500

@app.route('/stop_video', methods=['POST'])
def stop_video():
    """Detiene el procesamiento del video"""
    global video_capture, is_processing, processing_thread
    
    is_processing = False
    
    if processing_thread is not None:
        # Esperar a que el hilo termine (con timeout)
        if processing_thread.is_alive():
            processing_thread.join(timeout=1.0)
    
    if video_capture is not None:
        video_capture.release()
        video_capture = None
    
    return jsonify({"status": "success"})

@app.route('/get_latest_description')
def get_latest_description():
    """Retorna la última descripción generada"""
    return jsonify({"description": last_description, "is_processing": is_processing})

if __name__ == '__main__':
    import numpy as np  # Importación necesaria para el frame en blanco
    app.run(debug=True)