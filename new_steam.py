import paho.mqtt.client as mqtt
import threading
import pandas as pd
import pymongo
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import time
from twilio.rest import Client

# --- Streamlit Config ---
st.set_page_config(page_title="MQTT Live Sensor Dashboard", layout="centered")

# --- MongoDB Setup ---
MONGO_URI = "mongodb+srv://dharaneeshrajendran2004:LOPljtBufUwSBsJK@cluster0.uzi1r.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)
db = client["smartCityDB"]
collection = db["bookqs"]

# --- Twilio Setup ---
TWILIO_SID = "ACe3b1dc0fc96fc22bd462c8b934c2132c"
TWILIO_AUTH_TOKEN = "4ae6daf39da90df04e507e7e74a93c3e"
TWILIO_PHONE = "+18577887836"  # Twilio number like '+1234567890'

PHONE_NUMBERS = [
    "+916369586491"
]

def send_mass_alert():
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        message = "🚨 ALERT: High gas emissions detected in Coimbatore, Peelamedu for the last 5 readings. Immediate action required!"
        for number in PHONE_NUMBERS:
            client.messages.create(
                body=message,
                from_=TWILIO_PHONE,
                to=number
            )
            print(f"✅ Alert sent to {number}")
    except Exception as e:
        print("❌ Error sending Twilio alert:", e)

# --- Global Cache and Alert State ---
latest_data = {"Temperature": None, "Gas": None, "Threshold": None, "Analog": None}
danger_count = 0
alert_sent = False

# --- Train Model ---
def load_and_train_model():
    data = list(collection.find({}, {'_id': 0}))
    df = pd.DataFrame(data)

    if df.empty:
        return None, None

    df['Location'] = df.get('Location', 'Unknown').fillna("Unknown").astype(str)
    df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce').fillna(0)
    df['Gas Emission Value'] = pd.to_numeric(df['Gas Emission Value'], errors='coerce').fillna(0)
    df['Threshold'] = pd.to_numeric(df['Threshold'], errors='coerce').fillna(0)

    le = LabelEncoder()
    df['Location_Encoded'] = le.fit_transform(df['Location'])

    X = df[["Location_Encoded", "Temperature", "Gas Emission Value"]]
    y = df["Threshold"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le

@st.cache_resource(show_spinner=False)
def load_model():
    return load_and_train_model()

model, label_encoder = load_model()

# --- MQTT Setup ---
broker = "test.mosquitto.org"
port = 1883
topic = "esp32/sensor_data"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to MQTT Broker")
        client.subscribe(topic)
    else:
        print("❌ MQTT Connection failed with code", rc)

def on_message(client, userdata, msg):
    global danger_count, alert_sent

    try:
        payload = msg.payload.decode()
        temperature, gas = payload.split(",")
        temperature = float(temperature)
        gas = int(gas)

        location = "Coimbatore, Peelamedu"

        if model and label_encoder:
            if location not in label_encoder.classes_:
                label_encoder.classes_ = list(label_encoder.classes_) + [location]
            loc_encoded = label_encoder.transform([location])[0]

            features = pd.DataFrame([{
                "Location_Encoded": loc_encoded,
                "Temperature": temperature,
                "Gas Emission Value": gas
            }])
            threshold = model.predict(features)[0]
        else:
            threshold = gas * 1.1

        analog = 1 if gas > threshold else 0

        entry = {
            "Location": location,
            "Temperature": temperature,
            "Gas Emission Value": gas,
            "Threshold": round(float(threshold), 2),
            "Analog": analog,
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Date": datetime.now().date().isoformat()
        }

        collection.insert_one(entry)

        latest_data.update({
            "Temperature": temperature,
            "Gas": gas,
            "Threshold": round(float(threshold), 2),
            "Analog": analog
        })

        # Check for 5 consecutive danger readings
        if analog == 1:
            danger_count += 1
        else:
            danger_count = 0
            alert_sent = False

        if danger_count >= 5 and not alert_sent:
            send_mass_alert()
            alert_sent = True

        print(f"[MQTT] Data saved: {entry}")
    except Exception as e:
        print("❌ Error in on_message:", e)

def mqtt_thread():
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.connect(broker, port, 60)
    mqtt_client.loop_forever()

# --- Start MQTT Thread ---
threading.Thread(target=mqtt_thread, daemon=True).start()

# --- Streamlit UI ---
st.title("🔥 Environmental Sensor Dashboard")
st.markdown("Location: Coimbatore, Peelamedu")
st.divider()

# Display live data in real-time
placeholder = st.empty()

while True:
    with placeholder.container():
        st.subheader("📡 Latest Sensor Data")
        temp = latest_data["Temperature"]
        gas = latest_data["Gas"]
        threshold = latest_data["Threshold"]
        analog = latest_data["Analog"]

        if temp is not None:
            st.metric("🌡 Temperature (°C)", f"{temp:.2f}")
            st.metric("🫁 Gas Emission", f"{gas}")
            st.metric("📊 Threshold", f"{threshold}")
            st.metric("⚠ Analog Alert", "🔴 Danger" if analog else "🟢 Safe")
        else:
            st.warning("Waiting for live sensor data...")

    time.sleep(15)