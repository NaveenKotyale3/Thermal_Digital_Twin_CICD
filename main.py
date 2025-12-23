from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import logging
from collections import deque
import asyncio
import time
from typing import Dict      
import numpy as np
from sklearn.preprocessing import StandardScaler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from botocore.exceptions import ClientError
import uuid
import math

# Load .env 
load_dotenv()

app = FastAPI()

# ENV Constants
ZELIOT_BASE_URL = os.getenv("ZELIOT_BASE_URL")
ZELIOT_TOKEN = os.getenv("ZELIOT_TOKEN")
TRAKMATE_API = os.getenv("TRAKMATE_API")
TRAKMATE_APIKEY = os.getenv("TRAKMATE_APIKEY")


WINDOW_SIZE = 30  # Window size for both models
HORIZON = 30  # Prediction horizon (steps ahead)

# List of vehicle IDs to fetch data for 
def load_vehicle_ids():
    try:
        with open("vehicles.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("vehicles.json not found! Using fallback list.")
        raise HTTPException(status_code=500, detail="vehicles not found!")
       # fallback

VEHICLE_IDS = load_vehicle_ids()

# Expected features for the Random Forest model
EXPECTED_FEATURES = [  
    'SOC', 'SOH', 'Battery_pack_total_voltage', 'Battery_current',
    *[f'Battery_{i}_Volt' for i in range(1, 17)],
    *[f'temperature_{i}' for i in range(1, 7)]
]

# Thread-safe sliding window (per vehicle)
data_windows = {}  # Dictionary to store sliding windows for each vehicle_id
lock = asyncio.Lock()

# Preload models and scalers
MODEL_2_3KWH = joblib.load("rf_model_2_3kwh_window.pkl")
SCALER_2_3KWH = joblib.load("scaler_2_3kwh_window.pkl")
MODEL_3KWH = joblib.load("rf_model_3kwh_window.pkl")
SCALER_3KWH = joblib.load("scaler_3kwh_window.pkl")
                    
# Scheduler for automated data collection
scheduler = AsyncIOScheduler()  

# --- ✅ API Schema ---
class VehicleRequest(BaseModel):
    vehicle_id: str

class SensorData(BaseModel):
    vehicle_id: str
    timestamp: float = None
    soc: float = None   
    soh: float = None
    battery_pack_total_voltage: float = None
    battery_current: float = None
    battery_voltages: Dict = None
    temperatures: Dict = None
    other_features: Dict = None

# ----- ✅ Utility -----
def get_request_id():
    return str(uuid.uuid4())

def parse_timestamp(ts: str) -> datetime:
    if not ts:
        raise ValueError("Timestamp is missing") 
    try:  
        if ts.isdigit():
            return datetime.fromtimestamp(int(ts), tz=ZoneInfo("Asia/Kolkata"))
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ZoneInfo("Asia/Kolkata"))
    except Exception as e:
        raise ValueError(f"Invalid timestamp format: {ts} — {str(e)}")

def prepare_sliding_window_data(vehicle_id: str) -> np.ndarray: 
    """Prepare the last WINDOW_SIZE timesteps for prediction."""
    if vehicle_id not in data_windows or len(data_windows[vehicle_id]) < WINDOW_SIZE:
        raise HTTPException(status_code=400, detail=f"Insufficient data in sliding window for vehicle {vehicle_id}. Need {WINDOW_SIZE} points, got {len(data_windows.get(vehicle_id, []))}")

    window = list(data_windows[vehicle_id])[-WINDOW_SIZE:]  # Get the last WINDOW_SIZE data points
    window_data = []

    for data_point in window:
        row = []
        for field in ['soc', 'soh', 'battery_pack_total_voltage', 'battery_current']:
            value = data_point[field] if data_point[field] is not None else 0.0
            row.append(float(value))
        for i in range(1, 17):
            fname = f'Battery_{i}_Volt'
            value = data_point['battery_voltages'].get(fname, 0.0)
            row.append(float(value))
        for i in range(1, 7):
            fname = f'temperature_{i}'
            value = data_point['temperatures'].get(fname, 0.0)
            row.append(float(value)) 
        window_data.append(row)

    window_array = np.array(window_data).reshape(1, WINDOW_SIZE, len(EXPECTED_FEATURES))
    return window_array.reshape(1, -1)

async def fetch_and_post_sensor_data(vehicle_id: str):
    try:
        api_response = None
        packet_time = None
        full_capacity = 0

        if vehicle_id == "861557068891727":  # Zeliot
            url = f"{ZELIOT_BASE_URL}{vehicle_id}"
            headers = {"Authorization": f"Bearer {ZELIOT_TOKEN}"}
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code != 200:
                return
            packet = response.json()
            api_response = packet
            packet_time = parse_timestamp(packet.get("timestamp"))

            try:
                full_capacity = int(float(packet.get("full_capacity", 0)))
            except (ValueError, TypeError):
                full_capacity = 0

            input_dict = {
                "SOC": packet.get("soc"),
                "SOH": packet.get("soh"),
                "Battery_pack_total_voltage": packet.get("battery_pack_total_voltage"),
                "Battery_current": packet.get("battery_current")
            }
            for i in range(1, 17):
                input_dict[f"Battery_{i}_Volt"] = packet.get(f"battery_{i}_volt")
            for i in range(1, 7):
                input_dict[f"temperature_{i}"] = packet.get(f"temperature_{i}")

            if all(v is None for v in input_dict.values()):
                return
            if any(v is None for v in input_dict.values()):
                return

            input_dict = {
                k: float(str(v).replace("%", "").replace("V", "").replace("A", "").strip())
                for k, v in input_dict.items() if v is not None
            }
            battery_voltages = {f"Battery_{i}_Volt": input_dict.get(f"Battery_{i}_Volt") for i in range(1, 17)}
            temperatures = {f"temperature_{i}": input_dict.get(f"temperature_{i}") for i in range(1, 7)}
            sensor_data = {
                "vehicle_id": vehicle_id,
                "timestamp": packet_time.timestamp(),
                "soc": input_dict.get("SOC"),
                "soh": input_dict.get("SOH"),
                "battery_pack_total_voltage": input_dict.get("Battery_pack_total_voltage"),
                "battery_current": input_dict.get("Battery_current"),
                "battery_voltages": battery_voltages,
                "temperatures": temperatures,
                "other_features": {"full_capacity": full_capacity}
            }

        else:  # Trakmate
            url = f"{TRAKMATE_API}{vehicle_id}"
            headers = {"accept": "*/*", "apikey": TRAKMATE_APIKEY}
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code != 200:                         
                return
            entities = response.json().get("entities", [])
            bms_data = next((e for e in entities if e.get("name") == "E3_bms"), None)
            if not bms_data:
                return
            api_response = bms_data
            raw_ts = bms_data.get("timestamp") or bms_data.get("time") or bms_data.get("updatedAt")
            packet_time = parse_timestamp(raw_ts)

            try:
                full_capacity = int(float(bms_data.get("Full_Capacity (mAH)", 0)))
            except (ValueError, TypeError):
                full_capacity = 0

            input_dict = {
                "SOC": bms_data.get("SOC (%)"),
                "SOH": bms_data.get("SOH"),
                "Battery_pack_total_voltage": bms_data.get("Battery_pack_total_voltage (V)"),
                "Battery_current": bms_data.get("Battery_current (A)")
            }
            for i in range(1, 17):
                mV = bms_data.get(f"Battery_{i}_Volt (mV)")
                input_dict[f"Battery_{i}_Volt"] = float(mV) / 1000 if mV else None
            for i in range(1, 7):
                temp = bms_data.get(f"temperature_{i}")
                input_dict[f"temperature_{i}"] = float(temp) if temp else None

            if all(v is None for v in input_dict.values()):
                return
            if any(v is None for v in input_dict.values()):
                return

            input_dict = {
                k: float(str(v).replace("%", "").replace("V", "").replace("A", "").strip())
                for k, v in input_dict.items() if v is not None
            }
            battery_voltages = {f"Battery_{i}_Volt": input_dict.get(f"Battery_{i}_Volt") for i in range(1, 17)}
            temperatures = {f"temperature_{i}": input_dict.get(f"temperature_{i}") for i in range(1, 7)}
            sensor_data = {
                "vehicle_id": vehicle_id,
                "timestamp": packet_time.timestamp(), 
                "soc": input_dict.get("SOC"),
                "soh": input_dict.get("SOH"),
                "battery_pack_total_voltage": input_dict.get("Battery_pack_total_voltage"),
                "battery_current": input_dict.get("Battery_current"),
                "battery_voltages": battery_voltages,
                "temperatures": temperatures,
                "other_features": {"full_capacity": full_capacity}
            }

        async with lock:
            if vehicle_id not in data_windows:
                data_windows[vehicle_id] = deque(maxlen=WINDOW_SIZE)
            data_point = {
                "timestamp": sensor_data["timestamp"],
                "soc": sensor_data["soc"],
                "soh": sensor_data["soh"],
                "battery_pack_total_voltage": sensor_data["battery_pack_total_voltage"],
                "battery_current": sensor_data["battery_current"],
                "battery_voltages": sensor_data["battery_voltages"],
                "temperatures": sensor_data["temperatures"],
                "other_features": sensor_data["other_features"]
            }
            data_windows[vehicle_id].append(data_point)

    except Exception as e:
        pass  # Silently handle errors without logging

async def schedule_data_collection():
    tasks = [fetch_and_post_sensor_data(vehicle_id) for vehicle_id in VEHICLE_IDS]
    await asyncio.gather(*tasks)    

@app.on_event("startup")  
async def startup_event():
    scheduler.add_job(schedule_data_collection, IntervalTrigger(seconds=10))
    scheduler.start() 

@app.post("/sensor-data")
async def receive_sensor_data(data: SensorData):   
    vehicle_id = data.vehicle_id
    try:
        async with lock:
            if vehicle_id not in data_windows:
                data_windows[vehicle_id] = deque(maxlen=WINDOW_SIZE)

            for field in ['soc', 'soh', 'battery_pack_total_voltage', 'battery_current']:
                value = getattr(data, field)
                if value is None:
                    raise HTTPException(status_code=400, detail=f"Missing {field}")
                if not isinstance(value, (int, float)):
                    raise HTTPException(status_code=400, detail=f"Invalid {field}: must be a number")

            battery_voltages = data.battery_voltages or {}
            temperatures = data.temperatures or {}
            for i in range(1, 17):
                if f'Battery_{i}_Volt' not in battery_voltages or battery_voltages[f'Battery_{i}_Volt'] is None:
                    raise HTTPException(status_code=400, detail=f"Missing Battery_{i}_Volt")
            for i in range(1, 7):
                if f'temperature_{i}' not in temperatures or temperatures[f'temperature_{i}'] is None:
                    raise HTTPException(status_code=400, detail=f"Missing temperature_{i}")

            timestamp = data.timestamp if data.timestamp else time.time()
            data_point = {
                "timestamp": timestamp,
                "soc": data.soc,
                "soh": data.soh,
                "battery_pack_total_voltage": data.battery_pack_total_voltage,
                "battery_current": data.battery_current,
                "battery_voltages": battery_voltages,
                "temperatures": temperatures,
                "other_features": data.other_features or {}
            }
            data_windows[vehicle_id].append(data_point)

            return {
                "status": "success",
                "message": "Data received and stored",
                "vehicle_id": vehicle_id,
                "window_size": len(data_windows[vehicle_id])
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing sensor data: {str(e)}")
    
@app.get("/versions")
def get_versions():
    return {"versions":"v2"}

@app.get("/current-window/{vehicle_id}")
async def get_current_window(vehicle_id: str):
    async with lock:
        if vehicle_id not in data_windows:
            return {"status": "success", "vehicle_id": vehicle_id, "window_data": [], "window_size": 0}
        return {"status": "success", "vehicle_id": vehicle_id, "window_data": list(data_windows[vehicle_id]), "window_size": len(data_windows[vehicle_id])}

@app.post("/predict")
async def predict_temperature(request: VehicleRequest, http_request: Request):
    vehicle_id = request.vehicle_id
    client_ip = http_request.client.host
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    request_id = get_request_id()

    try:
        api_response = None
        packet_time = None
        full_capacity = 0

        # Fetch latest data to check vehicle status and freshness
        if vehicle_id == "861557068891727":  # Zeliot   
            url = f"{ZELIOT_BASE_URL}{vehicle_id}"
            headers = {"Authorization": f"Bearer {ZELIOT_TOKEN}"}
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Failed to fetch data from Zeliot API")
            packet = response.json()
            api_response = packet
            packet_time = parse_timestamp(packet.get("timestamp"))

            try:
                full_capacity = int(float(packet.get("full_capacity", 0)))
            except (ValueError, TypeError):
                full_capacity = 0         

            input_dict = {
                "SOC": packet.get("soc"),
                "SOH": packet.get("soh"),
                "Battery_pack_total_voltage": packet.get("battery_pack_total_voltage"),
                "Battery_current": packet.get("battery_current")
            }
            for i in range(1, 17):
                input_dict[f"Battery_{i}_Volt"] = packet.get(f"battery_{i}_volt")
            for i in range(1, 7):
                input_dict[f"temperature_{i}"] = packet.get(f"temperature_{i}")

            if all(v is None for v in input_dict.values()):
                raise HTTPException(status_code=400, detail=f"Vehicle {vehicle_id} is off or no data available.")
            if any(v is None for v in input_dict.values()):
                raise HTTPException(status_code=400, detail=f"Incomplete sensor data for {vehicle_id}.")

            input_dict = {
                k: float(str(v).replace("%", "").replace("V", "").replace("A", "").strip())
                for k, v in input_dict.items() if v is not None
            }
            battery_voltages = {f"Battery_{i}_Volt": input_dict.get(f"Battery_{i}_Volt") for i in range(1, 17)}
            temperatures = {f"temperature_{i}": input_dict.get(f"temperature_{i}") for i in range(1, 7)}
            sensor_data = {
                "vehicle_id": vehicle_id,
                "timestamp": packet_time.timestamp(),
                "soc": input_dict.get("SOC"),
                "soh": input_dict.get("SOH"),
                "battery_pack_total_voltage": input_dict.get("Battery_pack_total_voltage"),
                "battery_current": input_dict.get("Battery_current"),
                "battery_voltages": battery_voltages,
                "temperatures": temperatures,
                "other_features": {"full_capacity": full_capacity}
            }

        else:  # Trakmate
            url = f"{TRAKMATE_API}{vehicle_id}"
            headers = {"accept": "*/*", "apikey": TRAKMATE_APIKEY}
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Failed to fetch data from Trakmate API")
            entities = response.json().get("entities", [])
            bms_data = next((e for e in entities if e.get("name") == "E3_bms"), None)
            if not bms_data:
                raise HTTPException(status_code=500, detail=f"BMS data not found in Trakmate response")
            api_response = bms_data
            raw_ts = bms_data.get("timestamp") or bms_data.get("time") or bms_data.get("updatedAt")
            packet_time = parse_timestamp(raw_ts)

            try:
                full_capacity = int(float(bms_data.get("Full_Capacity (mAH)", 0)))
            except (ValueError, TypeError):
                full_capacity = 0

            input_dict = {
                "SOC": bms_data.get("SOC (%)"),
                "SOH": bms_data.get("SOH"),
                "Battery_pack_total_voltage": bms_data.get("Battery_pack_total_voltage (V)"),
                "Battery_current": bms_data.get("Battery_current (A)")
            }
            for i in range(1, 17):
                mV = bms_data.get(f"Battery_{i}_Volt (mV)")
                input_dict[f"Battery_{i}_Volt"] = float(mV) / 1000 if mV else None
            for i in range(1, 7):
                temp = bms_data.get(f"temperature_{i}")
                input_dict[f"temperature_{i}"] = float(temp) if temp else None

            if all(v is None for v in input_dict.values()):
                raise HTTPException(status_code=400, detail=f"Vehicle {vehicle_id} is off or no data available.")
            if any(v is None for v in input_dict.values()):
                raise HTTPException(status_code=400, detail=f"Incomplete sensor data for {vehicle_id}.")

            input_dict = {
                k: float(str(v).replace("%", "").replace("V", "").replace("A", "").strip())
                for k, v in input_dict.items() if v is not None 
            }
            battery_voltages = {f"Battery_{i}_Volt": input_dict.get(f"Battery_{i}_Volt") for i in range(1, 17)}
            temperatures = {f"temperature_{i}": input_dict.get(f"temperature_{i}") for i in range(1, 7)}
            sensor_data = {
                "vehicle_id": vehicle_id,
                "timestamp": packet_time.timestamp(),
                "soc": input_dict.get("SOC"),
                "soh": input_dict.get("SOH"),
                "battery_pack_total_voltage": input_dict.get("Battery_pack_total_voltage"),
                "battery_current": input_dict.get("Battery_current"),
                "battery_voltages": battery_voltages,
                "temperatures": temperatures,
                "other_features": {"full_capacity": full_capacity}
            }

        # Update sliding window with fresh data
        async with lock:
            if vehicle_id not in data_windows:
                data_windows[vehicle_id] = deque(maxlen=WINDOW_SIZE)
            data_point = {
                "timestamp": sensor_data["timestamp"],
                "soc": sensor_data["soc"],
                "soh": sensor_data["soh"],
                "battery_pack_total_voltage": sensor_data["battery_pack_total_voltage"],
                "battery_current": sensor_data["battery_current"],
                "battery_voltages": sensor_data["battery_voltages"],
                "temperatures": sensor_data["temperatures"],
                "other_features": sensor_data["other_features"]
            }
            data_windows[vehicle_id].append(data_point)

            # Check if window has enough data for prediction
            if len(data_windows[vehicle_id]) < WINDOW_SIZE:
                raise HTTPException(status_code=400, detail=f"Insufficient data for {vehicle_id}. Need {WINDOW_SIZE} points, got {len(data_windows[vehicle_id])}")

            input_array = prepare_sliding_window_data(vehicle_id)

        model = MODEL_2_3KWH if full_capacity < 50000 else MODEL_3KWH
        scaler = SCALER_2_3KWH if full_capacity < 50000 else SCALER_3KWH

        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)[0]
        prediction = [round(float(p), 2) for p in prediction]

        future_time = now_ist + timedelta(minutes=5)
        # Prepare input_log with plain floats rounded to 2 decimal places
        input_log = [
            {key: round(float(value), 2) for key, value in zip(EXPECTED_FEATURES, row)}
            for row in input_array.reshape(WINDOW_SIZE, len(EXPECTED_FEATURES))
        ]
        # Prepare pred_result with only temperature predictions
        pred_result = {
            "BMS Temperature": prediction[0],
            "Battery Pack Temperature 1": prediction[1],
            "Battery Pack Temperature 2": prediction[2],
            "Battery Pack Temperature 3": prediction[3],
            "Battery Pack Temperature 4": prediction[4]
        }

        # Return response including input_window_data as required
        response_pred_result = {
            "BMS Temperature": prediction[0],
            "Battery Pack Temperature 1": prediction[1],
            "Battery Pack Temperature 2": prediction[2],
            "Battery Pack Temperature 3": prediction[3],
            "Battery Pack Temperature 4": prediction[4],
            "input_window_data": input_log
        }

        return {
            "vehicle_id": vehicle_id,
            "prediction": response_pred_result,
            "packet_timestamp": packet_time.strftime("%Y-%m-%d %H:%M:%S"),
            "prediction_for": future_time.strftime("%Y-%m-%d %H:%M:%S"),
            "window_size": len(data_windows.get(vehicle_id, []))
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")