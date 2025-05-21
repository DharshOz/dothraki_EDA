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
db = client["dL"]
collection = db["Bookq"]

# --- Twilio Setup ---
TWILIO_SID = "ACe3b1dc0fc96fc22bd462c8b934c2132c"
TWILIO_AUTH_TOKEN = "4ae6daf39da90df04e507e7e74a93c3e"
TWILIO_PHONE = "+18577887836"

PHONE_NUMBERS = ["+916369586491"]

def send_mass_alert():
    global latest_data
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        message = "🚨 ALERT: High gas emissions detected in Coimbatore, Peelamedu for the last 5 readings. Immediate action required!"
        
        # Track successful sends
        successful_sends = 0
        failed_numbers = []
        
        for number in PHONE_NUMBERS:
            try:
                client.messages.create(
                    body=message,
                    from_=TWILIO_PHONE,
                    to=number
                )
                successful_sends += 1
                print(f"✅ Alert successfully sent to {number}")
            except Exception as e:
                failed_numbers.append(number)
                print(f"❌ Failed to send alert to {number}: {str(e)}")
        
        # Update latest_data with alert status
        latest_data["Alert"] = {
            "Status": "Sent" if successful_sends > 0 else "Failed",
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Successful": successful_sends,
            "Failed": len(failed_numbers),
            "FailedNumbers": failed_numbers
        }
        
        return successful_sends > 0
        
    except Exception as e:
        print("❌ Error in Twilio client setup:", e)
        latest_data["Alert"] = {
            "Status": "Error",
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Error": str(e)
        }
        return False

# --- Global Cache and Alert State ---
latest_data = {
    "Temperature": None, 
    "Gas": None, 
    "Threshold": None, 
    "Analog": None, 
    "Stored": False,
    "Alert": None,
    "DangerCount": 0
}
danger_count = 0
alert_sent = False

# --- Train Model ---
def load_and_train_model():
    try:
        data = list(collection.find({}, {'_id': 0}))
        df = pd.DataFrame(data)

        if df.empty:
            print("⚠️ No data found in database for training")
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

        print("✅ Model trained successfully")
        return model, le
    except Exception as e:
        print("❌ Model training failed:", e)
        return None, None

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
        print(f"❌ MQTT Connection failed with code {rc}")

def on_message(client, userdata, msg):
    global danger_count, alert_sent, latest_data

    try:
        payload = msg.payload.decode()
        print(f"📩 Received raw MQTT data: {payload}")
        
        temperature, gas = payload.split(",")
        temperature = float(temperature)
        gas = int(gas)
        
        print(f"🌡️ Temperature: {temperature}°C | 💨 Gas: {gas}")

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
            print(f"🧠 Model predicted threshold: {threshold:.2f}")
        else:
            threshold = gas * 1.1
            print(f"⚠️ Using fallback threshold: {threshold:.2f}")

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

        # Store in MongoDB
        try:
            collection.insert_one(entry)
            print(f"💾 Stored in DB: {entry}")
            latest_data["Stored"] = True
        except Exception as e:
            print(f"❌ DB storage failed: {e}")
            latest_data["Stored"] = False

        latest_data.update({
            "Temperature": temperature,
            "Gas": gas,
            "Threshold": round(float(threshold), 2),
            "Analog": analog,
            "DangerCount": danger_count
        })

        # Check for 5 consecutive danger readings
        if analog == 1:
            danger_count += 1
            print(f"⚠️ Danger count: {danger_count}/5")
        else:
            danger_count = 0
            alert_sent = False

        if danger_count >= 1 and not alert_sent:
            print("🚨 Attempting to send alerts...")
            success = send_mass_alert()
            if success:
                print("✅ All alerts sent successfully!")
            alert_sent = True

    except Exception as e:
        print(f"❌ Error processing message: {e}")

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
        st.subheader("📡 Live Sensor Data")
        
        temp = latest_data["Temperature"]
        gas = latest_data["Gas"]
        threshold = latest_data["Threshold"]
        analog = latest_data["Analog"]
        stored = latest_data["Stored"]
        alert_data = latest_data.get("Alert", {})
        danger_count = latest_data.get("DangerCount", 0)

        if temp is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🌡 Temperature (°C)", f"{temp:.2f}")
                st.metric("🫁 Gas Emission", f"{gas}")
                
            with col2:
                st.metric("📊 Threshold", f"{threshold:.2f}")
                status = st.container()
                if analog:
                    status.error("⚠️ DANGER: Gas above threshold!")
                else:
                    status.success("✅ SAFE: Normal levels")
            
            # Alert status section
            st.subheader("🚨 Alert Status")
            alert_col1, alert_col2 = st.columns(2)
            
            with alert_col1:
                st.metric("Danger Count", f"{danger_count}/2")
                
            with alert_col2:
                if alert_data:
                    if alert_data.get("Status") == "Sent":
                        st.success(f"✅ Alert sent at {alert_data.get('Timestamp')}")
                        st.write(f"Successful: {alert_data.get('Successful')} | Failed: {alert_data.get('Failed')}")
                    elif alert_data.get("Status") == "Failed":
                        st.error(f"❌ Alert failed at {alert_data.get('Timestamp')}")
                        if alert_data.get("FailedNumbers"):
                            st.write(f"Failed numbers: {', '.join(alert_data.get('FailedNumbers'))}")
                    else:
                        st.warning("⚠️ Alert status unknown")
                else:
                    st.info("ℹ️ No alerts sent yet")
                
            st.caption(f"📦 Last reading stored in DB: {'✅ Yes' if stored else '❌ No'}")
            
            # Raw data display
            with st.expander("📜 Detailed Status"):
                st.json(latest_data)
        else:
            st.warning("Waiting for live sensor data...")
            
        st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

    time.sleep(10)