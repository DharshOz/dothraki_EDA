import multiprocessing
import cv2
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

MONGO_URI = "mongodb+srv://dharaneeshrajendran2004:LOPljtBufUwSBsJK@cluster0.uzi1r.mongodb.net/smartCityDB?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "smartCityDB"
READINGS_COLLECTION = "Bookq"
MEAN_VALUES_COLLECTION = "mean_json1"

client = pymongo.MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
readings_collection = db[READINGS_COLLECTION]
mean_values_collection = db[MEAN_VALUES_COLLECTION]

def webcam_feed(stop_signal):
    cap = cv2.VideoCapture(0)  
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Webcam Feed", frame)

        # Press 'q' to quit the webcam OR stop signal received
        if cv2.waitKey(1) & 0xFF == ord('q') or not stop_signal.empty():
            print("Stopping webcam feed...")
            break
    
    cap.release()
    cv2.destroyAllWindows()

def load_or_initialize_data():
    """Load data from MongoDB"""
    cursor = readings_collection.find({}, {'_id': 0})  
    data = list(cursor)
    
    if data:
        df = pd.DataFrame(data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp'])  
        df['Date'] = df['Timestamp'].dt.date.astype(str)  
    else:
        df = pd.DataFrame(columns=[
            "Location", "Temperature", "Gas Emission Value",
            "Threshold", "Analog", "Timestamp", "Date"
        ])
    
    df['Location'] = df['Location'].fillna("Unknown")
    return df

def handle_location_encoding(df):
    """Create and fit label encoder with all locations"""
    label_encoder = LabelEncoder()
    if not df.empty and 'Location' in df.columns:
        unique_locations = df['Location'].unique().tolist()
        if "Unknown" not in unique_locations:
            unique_locations.append("Unknown")
        label_encoder.fit(unique_locations)
    return label_encoder
def save_mean_values():
    """Calculate and store daily mean values in MongoDB"""
    df = load_or_initialize_data()
    
    if not df.empty:
        mean_values = df.groupby('Date').agg({
            'Temperature':'mean',
            'Threshold': 'mean',
            'Gas Emission Value': 'mean'
        }).reset_index()
        
        mean_values_records = mean_values.to_dict(orient='records')
        
        # Replace existing mean values for the same date
        for record in mean_values_records:
            mean_values_collection.update_one(
                {"Date": record["Date"]}, 
                {"$set": record}, 
                upsert=True
            )

        print("\nDaily mean values saved to MongoDB successfully!")

def train_model(df, label_encoder):
    """Train Random Forest model if possible"""
    if "Threshold" in df.columns and not df.empty and 'Location' in df.columns:
        try:
            df["Location_Encoded"] = label_encoder.transform(df["Location"])
            
            X = df[["Location_Encoded", "Temperature", "Gas Emission Value"]]
            y = df["Threshold"]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Calculate metrics
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            print(f"Model Metrics:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}")
            
            return model
            
        except ValueError as e:
            print(f"Model training error: {e}")
            return None
    
    print("Insufficient data for model training")
    return None

def get_user_input():
    """Collect user input in the main process"""
    location = input("Enter Location: ").strip() or "Unknown"
    try:
        temperature = float(input("Enter Temperature: "))
        gas_emission = float(input("Enter Gas Emission Value: "))
        return location, temperature, gas_emission
    except ValueError:
        print("Invalid input! Please enter numeric values for temperature and gas emission.")
        return get_user_input()

def process_user_input():
    """Handle user input and process predictions"""
    df = load_or_initialize_data()
    label_encoder = handle_location_encoding(df)
    model = train_model(df, label_encoder)

    while True:
        location, temperature, gas_emission = get_user_input()

        # Handle location encoding
        try:
            location_encoded = label_encoder.transform([location])[0]
        except ValueError:
            new_classes = list(label_encoder.classes_) + [location]
            label_encoder.fit(new_classes)
            location_encoded = label_encoder.transform([location])[0]

        # Make prediction
        if model:
            user_input = pd.DataFrame([{
                "Location_Encoded": location_encoded,
                "Temperature": temperature,
                "Gas Emission Value": gas_emission
            }])
            predicted_threshold = model.predict(user_input)[0]
        else:
            predicted_threshold = gas_emission * 1.1  # Default logic

        analog = 1 if gas_emission > predicted_threshold else 0

        print(f"\nPrediction Results:")
        print(f"Threshold: {predicted_threshold:.2f}")
        print(f"Analog Value: {analog}")

        # Create new entry with timestamp
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_entry = {
            "Location": location,
            "Temperature": temperature,
            "Gas Emission Value": gas_emission,
            "Threshold": float(predicted_threshold),
            "Analog": int(analog),
            "Timestamp": current_timestamp,
            "Date": datetime.now().date().isoformat()
        }

        # Save new data to MongoDB
        readings_collection.insert_one(new_entry)
        print("\nData saved to MongoDB successfully!")
        save_mean_values()

        # Ask user if they want to stop
        stop_choice = input("Type 'exit' to stop, or press Enter to continue: ").strip().lower()
        if stop_choice == "exit":
            return True  # Stop main process
import cv2
import torch
import cloudinary
import cloudinary.uploader
import threading
import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pymongo import MongoClient
from datetime import datetime, timezone

def initialize_device():
    """Initialize the device (GPU or CPU)"""
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_model(device):
    """Load the CLIP model and processor"""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    return model, processor

def get_class_names():
    """Return predefined class names for accident detection"""
    return [
         "injury", "hand", "police", "car crash", "collision",
        "accident", "roadblock", "damaged vehicle", "broken glass", "explosion", "smoke","fire"
    ]

def configure_cloudinary():
    """Configure Cloudinary settings"""
    cloudinary.config(
        cloud_name="ddv0mpecp",
        api_key="396472696738852",
        api_secret="5E2ykB-yCWxSPFKFzFTlo8Sxmac"
    )

def connect_mongodb():
    """Connect to MongoDB and return the collection"""
    client = MongoClient("mongodb+srv://dharaneeshrajendran2004:LOPljtBufUwSBsJK@cluster0.uzi1r.mongodb.net/smartCityDB?retryWrites=true&w=majority&appName=Cluster0")
    db = client["smartCityDB"]
    return db["videos"]

def upload_file(file_path, collection, file_type="video"):
    """Upload video to Cloudinary and store the URL in MongoDB (with local file cleanup)"""
    try:
        response = cloudinary.uploader.upload(file_path, resource_type=file_type)
        video_url = response['secure_url']
        timestamp = datetime.now(timezone.utc)

        # Store in MongoDB
        collection.insert_one({"video_url": video_url, "timestamp": timestamp})
        print(f"Upload successful! URL: {video_url}")
        # Cleanup local file
        os.remove(file_path)
        return video_url
    except Exception as e:
        print(f"Upload failed: {e}")
        return None

def process_frame(frame, model, processor, class_names, device):
    """Process a frame through CLIP model and return detected objects"""
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_inputs = processor(images=image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs).half()
        text_inputs = processor(text=class_names, return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**text_inputs).half()
        similarity = (image_features @ text_features.T).softmax(dim=-1)

    sorted_indices = similarity[0].argsort(descending=True)
    detected_objects = [(class_names[idx], similarity[0][idx].item()) for idx in sorted_indices[:5]]

    return detected_objects

def draw_labels(frame, detected_objects):
    """Draw detected object labels on the frame"""
    y_offset = 30
    for obj, confidence in detected_objects:
        label = f"{obj}: {confidence:.2f}"
        cv2.putText(frame, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30

def accident_detection():
    """Main function to start accident detection with optimized uploads"""
    device = initialize_device()
    model, processor = load_clip_model(device)
    class_names = get_class_names()
    configure_cloudinary()
    collection = connect_mongodb()

    cap = cv2.VideoCapture(0)
    
    # Video parameters for smaller file size
    video_width, video_height = 640, 480
    frame_rate = 15  # Reduced from 20 for smaller files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec

    recording = False
    video_writer = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detected_objects = process_frame(frame, model, processor, class_names, device)
            draw_labels(frame, detected_objects)

            # Start recording if high confidence detected
            if any(confidence >= 0.60 for _, confidence in detected_objects):
                if not recording:
                    print("High-confidence event detected! Recording started.")
                    # Generate unique filename with timestamp
                    video_filename = f"accident_clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    video_writer = cv2.VideoWriter(video_filename, fourcc, frame_rate, (video_width, video_height))
                    recording = True

            # Process recording
            if recording:
                # Resize frame before writing to reduce file size
                resized_frame = cv2.resize(frame, (video_width, video_height))
                video_writer.write(resized_frame)
                
                # Stop recording if confidence drops
                if all(confidence < 0.50 for _, confidence in detected_objects):
                    print("Confidence dropped. Stopping recording and uploading...")
                    recording = False
                    video_writer.release()
                    video_writer = None
                    # Start upload in a background thread
                    upload_thread = threading.Thread(
                        target=upload_file,
                        args=(video_filename, collection)
                    )
                    upload_thread.start()

            cv2.imshow("Accident Detection - Press 'q' to exit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Program stopped by user.")
    finally:
        # Cleanup
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Start the accident detection process using multiprocessing
    accident_process = multiprocessing.Process(target=accident_detection)
    accident_process.start()

    # Run the user input process in the main process
    should_stop = process_user_input()

    # When the user opts to exit, terminate the accident detection process
    if should_stop:
        accident_process.terminate()  # Forcefully terminates the accident detection process
        accident_process.join()       # Wait for the process to properly end
        