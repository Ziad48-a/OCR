from fastapi import FastAPI, File, UploadFile  
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
import torch

from ocr_inference import load_model
from ocr_inference import preprocess
from ocr_inference import predict
import os

app = FastAPI(title="OCR API", description="Upload an image of a handwritten letter and get the predicted character.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
index_path = os.path.join(os.path.dirname(__file__), 'index.html')
model = None
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_mobilenet.pth')
device = "cuda" if torch.cuda.is_available() else "cpu"
@app.on_event("startup")
def startup_event():
    # if not os.path.exists(model_path):
    #     raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure it is in the correct location.")
    #     return
    global model
    model = load_model(model_path)
    model.to(device)
    print("✓ Model loaded successfully!")
@app.get("/health")
def health_check():
    return {"status": "OK",
            "message": "OCR API is running! ,"
            " model loaded successfully! , "
            "supported classes: " + ", ".join(CLASSES)}

#we need web dashboard

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")
    try:
        # Save the uploaded file to a temporary location
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Preprocess the image and make a prediction
        input_tensor, stages = preprocess(temp_file_path)  
        input_tensor = input_tensor.to(device)
        predicted_class, confidence = predict(model, input_tensor)
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        return {"predicted_character": predicted_class,
                "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open(index_path, 'r', encoding='utf-8') as f:  
        return f.read()
# Run the app with: python -m uvicorn main:app --reload