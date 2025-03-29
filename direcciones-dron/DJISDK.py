from djitellopy import Tello
import time

# 1. Conectar y despegar
tello = Tello()
tello.connect()  # Conecta al WiFi del Tello
tello.takeoff()

# 2. Definir waypoints (movimientos relativos en cm)
waypoints = [
    {"x": 100, "y": 0, "z": 50},   # Avanza 1m, sube 0.5m
    {"x": 0, "y": 100, "z": 0},    # Derecha 1m
    {"x": -100, "y": 0, "z": 0},   # Retrocede 1m
    {"x": 0, "y": -100, "z": -50}  # Izquierda 1m, baja 0.5m
]

# 3. Ejecutar ruta
for wp in waypoints:
    tello.go_xyz_speed(wp["x"], wp["y"], wp["z"], speed=30)  # Velocidad 30 cm/s
    time.sleep(2)  # Espera 2 segundos entre movimientos

# 4. Aterrizar
tello.land()