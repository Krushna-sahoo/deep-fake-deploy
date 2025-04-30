import torch
import cv2
import timm
import numpy as np
import tempfile
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
import torch.nn as nn
from io import BytesIO

# =======================
# 📌 Initialize FastAPI
# =======================
app = FastAPI()

# =======================
# 📌 Define Constants
# =======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_SIZE = (224, 224)
FRAMES_PER_VIDEO = 30

# =======================
# 📌 Define Model
# =======================
class DeepFakeDetector(nn.Module):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        self.feature_extractor = timm.create_model("efficientnet_lite0", pretrained=False)
        self.feature_extractor.classifier = nn.Identity()
        self.lstm = nn.LSTM(input_size=1280, hidden_size=256, num_layers=1, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(B, T, -1)
        lstm_out, _ = self.lstm(features)
        output = self.fc(lstm_out[:, -1, :])
        return output

# Load model
MODEL_PATH = r"C:\Users\KIIT\Documents\Minor\best_deepfake_detector2.pth"  # Update with correct path
model = DeepFakeDetector().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =======================
# 📌 Preprocessing Function
# =======================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_video(video_path, frames_per_video=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < frames_per_video and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)
    
    cap.release()

    while len(frames) < frames_per_video:
        frames.append(torch.zeros(3, *FRAME_SIZE))

    return torch.stack(frames).unsqueeze(0)  # Shape: (1, T, C, H, W)

# =======================
# 📌 Inference Endpoint
# =======================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(await file.read())
            temp_video_path = temp_video.name  # Get the temp file path

        # Preprocess the video
        video_tensor = preprocess_video(temp_video_path).to(DEVICE)

        # Run inference
        with torch.no_grad():
            output = model(video_tensor)
            pred = torch.argmax(output, dim=1).item()
            fake_prob = torch.softmax(output, dim=1)[0, 1].item()

        label = "FAKE" if pred == 1 else "REAL"
        
        return {"prediction": label, "fake_probability": fake_prob}

    except Exception as e:
        return {"error": str(e)}

# =======================
# 📌 Run FastAPI (Optional)
# =======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
