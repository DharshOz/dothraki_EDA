const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const mongoose = require('mongoose');
const cors = require('cors');
const { Client } = require('twilio');
const mqtt = require('mqtt');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, { cors: { origin: "*" } });

app.use(cors());
app.use(express.json());

// MongoDB connection
mongoose.connect("mongodb+srv://dharaneeshrajendran2004:LOPljtBufUwSBsJK@cluster0.uzi1r.mongodb.net/dL?retryWrites=true&w=majority");

const sensorSchema = new mongoose.Schema({
  Location: String,
  Temperature: Number,
  "Gas Emission Value": Number,
  Threshold: Number,
  Analog: Number,
  Timestamp: String,
  Date: String,
});

const Sensor = mongoose.model("Sensor", sensorSchema);

// Twilio configuration
const twilioClient = require('twilio')("ACe3b1dc0fc96fc22bd462c8b934c2132c", "4ae6daf39da90df04e507e7e74a93c3e");
const PHONE_NUMBERS = ["+916369586491"];

let dangerCount = 0;
let alertSent = false;

// MQTT connection
const mqttClient = mqtt.connect("mqtt://test.mosquitto.org");

mqttClient.on("connect", () => {
  console.log("✅ Connected to MQTT Broker");
  mqttClient.subscribe("esp32/sensor_data");
});

mqttClient.on("message", async (topic, message) => {
  const payload = message.toString();
  const [tempStr, gasStr] = payload.split(",");
  const Temperature = parseFloat(tempStr);
  const Gas = parseInt(gasStr);

  const Threshold = Gas * 1.1;
  const Analog = Gas > Threshold ? 1 : 0;
  const now = new Date();
  const Timestamp = now.toISOString();
  const DateOnly = now.toISOString().split("T")[0];

  const data = {
    Location: "Coimbatore, Peelamedu",
    Temperature,
    "Gas Emission Value": Gas,
    Threshold: parseFloat(Threshold.toFixed(2)),
    Analog,
    Timestamp,
    Date: DateOnly,
  };

  try {
    await Sensor.create(data);
    console.log("📦 Stored:", data);
  } catch (e) {
    console.error("❌ MongoDB error:", e.message);
  }

  io.emit("sensorData", data);

  if (Analog === 1) dangerCount++;
  else {
    dangerCount = 0;
    alertSent = false;
  }

  if (dangerCount >= 5 && !alertSent) {
    for (const number of PHONE_NUMBERS) {
      try {
        await twilioClient.messages.create({
          body: "🚨 ALERT: High gas emissions in Coimbatore, Peelamedu for last 5 readings.",
          from: "+18577887836",
          to: number,
        });
        console.log(`📞 Alert sent to ${number}`);
      } catch (e) {
        console.error(`❌ Failed alert to ${number}: ${e.message}`);
      }
    }
    alertSent = true;
  }
});

// Start server
server.listen(3000, () => {
  console.log("🌐 Server listening on http://localhost:3000");
});
