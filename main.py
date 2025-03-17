import pygame
import firebase_admin
from firebase_admin import credentials, db
import requests
from datetime import datetime
import threading
import pandas as pd
import random
import time

# Initialize Firebase
cred = credentials.Certificate("/mnt/d/DLprojects/Traffic_Handler_RL/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://traffic-handling-rl-default-rtdb.firebaseio.com/'
})

# Constants
TOMTOM_API_KEY = "G8DD3ESV5VYPl7zCM2c2adZuJZ8Ow2Fm"
STATE_SPACE = ["Low", "Medium", "High"]
ACTION_SPACE = ["Increase", "Decrease", "Keep Same"]
Q_TABLE = {}
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.2  # Exploration rate

# Pygame Initialization
pygame.init()
WIDTH, HEIGHT = 600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Traffic RL Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)

# Traffic Light Positions (Intersection Layout)
LIGHT_POSITIONS = [(250, 200), (350, 200), (250, 400), (350, 400)]
LIGHT_STATES = ["Red", "Red", "Red", "Red"]  # Default all red

def fetch_traffic_lights():
    lat_ref = db.reference("lat")
    lon_ref = db.reference("lon")
    latitudes = lat_ref.get()
    longitudes = lon_ref.get()
    return [{"lat": lat, "lon": lon} for lat, lon in zip(latitudes, longitudes)] if latitudes and longitudes else []

def get_traffic_data(location):
    lat, lon = location["lat"], location["lon"]
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative/10/json?point={lat},{lon}&key={TOMTOM_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if "flowSegmentData" in data:
            return {
                "latitude": lat,
                "longitude": lon,
                "current_speed": data["flowSegmentData"].get("currentSpeed", None),
                "free_flow_speed": data["flowSegmentData"].get("freeFlowSpeed", None),
                "congestion_level": classify_congestion(
                    data["flowSegmentData"].get("currentSpeed", None),
                    data["flowSegmentData"].get("freeFlowSpeed", None)
                ),
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        print(f"Error fetching data: {e}")
    return None

def classify_congestion(current_speed, free_flow_speed):
    if current_speed is None or free_flow_speed is None:
        return "Unknown"
    congestion_index = 1 - (current_speed / free_flow_speed)
    return "Low" if congestion_index < 0.3 else "Medium" if congestion_index < 0.7 else "High"

def store_traffic_data(data):
    if data:
        db.reference("traffic_data").push(data)

def get_best_action(state):
    if state not in Q_TABLE:
        Q_TABLE[state] = {action: 0 for action in ACTION_SPACE}
    return random.choice(ACTION_SPACE) if random.uniform(0, 1) < EPSILON else max(Q_TABLE[state], key=Q_TABLE[state].get)

def update_q_table(state, action, reward, next_state):
    if state not in Q_TABLE:
        Q_TABLE[state] = {action: 0 for action in ACTION_SPACE}
    if next_state not in Q_TABLE:
        Q_TABLE[next_state] = {action: 0 for action in ACTION_SPACE}
    old_value = Q_TABLE[state][action]
    future_max = max(Q_TABLE[next_state].values())
    Q_TABLE[state][action] = (1 - LEARNING_RATE) * old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * future_max)

def apply_rl_traffic_control():
    print("\n---- New RL Iteration ----")
    traffic_data = db.reference("traffic_data").get()
    if not traffic_data:
        print("No traffic data available.")
        return
    for idx, (key, row) in enumerate(traffic_data.items()):
        state = row["congestion_level"]
        action = get_best_action(state)
        reward = 1 if action == "Decrease" and state == "High" else -1 if action == "Increase" and state == "High" else 0
        next_state = "Low" if reward > 0 else "High"
        update_q_table(state, action, reward, next_state)
        LIGHT_STATES[idx % 4] = "Green" if action == "Decrease" else "Red"
        print(f"Traffic Light {idx % 4} - State: {state}, Action: {action}, Reward: {reward}")

def collect_and_store_data():
    locations = fetch_traffic_lights()
    threads = [threading.Thread(target=lambda: store_traffic_data(get_traffic_data(loc))) for loc in locations]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    apply_rl_traffic_control()

def draw_traffic_lights():
    screen.fill(WHITE)

    # Draw pole (black rectangle)
    pygame.draw.rect(screen, BLACK, (280, 100, 40, 300))

    # Draw light box (black)
    pygame.draw.rect(screen, BLACK, (260, 50, 80, 130))

    # Draw red, yellow, and green lights
    pygame.draw.circle(screen, RED, (300, 80), 20)     # Red light
    pygame.draw.circle(screen, YELLOW, (300, 120), 20)  # Yellow light
    pygame.draw.circle(screen, GREEN, (300, 160), 20)   # Green light

    pygame.display.flip()

def start_simulation():
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        draw_traffic_lights()
        clock.tick(2)  # 2 FPS refresh rate
    pygame.quit()

# Start Simulation Thread
sim_thread = threading.Thread(target=start_simulation)
sim_thread.start()

# Start Data Collection Loop
while True:
    collect_and_store_data()
    time.sleep(10)  # Fetch new data every 10 seconds
