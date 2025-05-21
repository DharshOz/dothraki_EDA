import cv2
import torch
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cloudinary
import cloudinary.uploader
from pymongo import MongoClient
from datetime import datetime, timezone

# MongoDB and Cloudinary Setup
cloudinary.config(
    cloud_name="ddv0mpecp",
    api_key="396472696738852",
    api_secret="5E2ykB-yCWxSPFKFzFTlo8Sxmac"
)

client = MongoClient("mongodb+srv://dharaneeshrajendran2004:LOPljtBufUwSBsJK@cluster0.uzi1r.mongodb.net/smartCityDB?retryWrites=true&w=majority&appName=Cluster0")
video_collection = client["smartCityDB"]["videos"]

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

class_names = [
    "injury", "hand", "police", "car crash", "collision",
    "accident", "roadblock", "damaged vehicle", "broken glass", "explosion", "smoke", "fire"
]

def detect_objects(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_feat = model.get_image_features(**inputs).half()
        text_inputs = processor(text=class_names, return_tensors="pt", padding=True).to(device)
        text_feat = model.get_text_features(**text_inputs).half()
        sim = (image_feat @ text_feat.T).softmax(dim=-1)

    best_match = sim[0].max().item()
    label = class_names[sim[0].argmax().item()]
    return label, best_match

def draw_label(frame, label, confidence):
    text = f"{label}: {confidence:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def upload_video(file_path):
    try:
        mp4_path = file_path.replace(".avi", ".mp4")

        # Convert AVI to MP4 using OpenCV
        cap = cv2.VideoCapture(file_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(mp4_path, fourcc, 15.0, (640, 480))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()

        # Upload and transcode properly on Cloudinary
        result = cloudinary.uploader.upload(
            mp4_path,
            resource_type="video",
            format="mp4",
            eager=[{
                "quality": "auto",
                "format": "mp4"
            }]
        )
        video_url = result['secure_url']

        # Store in MongoDB
        video_collection.insert_one({
            "video_url": video_url,
            "timestamp": datetime.now(timezone.utc)
        })

        print(f"[Uploaded] {video_url}")

        # Remove both local files
        os.remove(file_path)
        os.remove(mp4_path)

    except Exception as e:
        print(f"Upload failed: {e}")

def record_video():
    cap = cv2.VideoCapture(0)
    width, height = 640, 480
    frame_rate = 15
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    recording = False
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, confidence = detect_objects(frame)
        draw_label(frame, label, confidence)
        cv2.imshow("Accident Detection", frame)

        if confidence >= 0.7 and not recording:
            filename = f"accident_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            out = cv2.VideoWriter(filename, fourcc, frame_rate, (width, height))
            recording = True
            print("[Started Recording]")

        if recording:
            out.write(cv2.resize(frame, (width, height)))

        if recording and confidence < 0.4:
            recording = False
            out.release()
            print("[Stopped Recording] Uploading...")
            upload_video(filename)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if recording:
        out.release()
        upload_video(filename)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    record_video()
