import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import uvicorn
import base64
import numpy as np
import torch
import traceback

MODEL_PATH = "best.pt"
CLASS_NAMES_CONFIG = ['text', 'metal']
STATIC_DIR = "frontend"

app = FastAPI(title="MetalText YOLOv8 Detection API")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load YOLOv8 Model ---
model = None
model_device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ... (model loading logic from previous version - keep it the same) ...
try:
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Model file not found at '{os.getcwd()}/{MODEL_PATH}'.")
    else:
        model = YOLO(MODEL_PATH)
        model.to(model_device)
        print(f"INFO: Successfully loaded YOLOv8 model from {MODEL_PATH} to {model_device}")
        print(f"INFO: Model class names (from model file): {model.names}")
except Exception as e:
    print(f"CRITICAL ERROR: Error loading YOLOv8 model: {e}")
    traceback.print_exc()

@app.on_event("startup")
async def startup_event():
    if model is None:
        print("WARNING: YOLO Model was not loaded. The /predict endpoint will not work.")
    else:
        print("INFO: YOLO Model loaded. API is ready.")

# --- API Endpoint for Prediction (remains the same) ---
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # ... (Keep the entire /predict endpoint logic exactly as in the previous response) ...
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to load. API unavailable.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))
        original_mode = img.mode
        if original_mode not in ['RGB', 'RGBA']:
            if original_mode in ['P', 'LA', 'L']: img = img.convert('RGBA')
            else: img = img.convert('RGB')
        results = model.predict(source=img, device=model_device)
        predictions_data = []
        annotated_img = img.copy()
        if annotated_img.mode != 'RGBA': annotated_img = annotated_img.convert('RGBA')
        draw = ImageDraw.Draw(annotated_img)
        font_size_on_image = 24
        try: font = ImageFont.truetype("arial.ttf", font_size_on_image)
        except IOError:
            try: font = ImageFont.truetype("DejaVuSans.ttf", font_size_on_image)
            except IOError: font = ImageFont.load_default()
        if results and len(results) > 0:
            result = results[0]
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                class_name_to_display = model.names.get(cls_id, f"ID_{cls_id}") if model.names else CLASS_NAMES_CONFIG[cls_id] if 0 <= cls_id < len(CLASS_NAMES_CONFIG) else f"ID_{cls_id}"
                predictions_data.append({"class_id": cls_id, "class_name": class_name_to_display, "confidence": round(conf, 2), "bbox": [x1, y1, x2, y2]})
                conf_percentage_on_image = int(conf * 100)
                text_label_on_image = f"{class_name_to_display} {conf_percentage_on_image}%"
                text_w, text_h = 0,0
                try:
                    if hasattr(draw, 'textlength'):
                        text_w = draw.textlength(text_label_on_image, font=font)
                        if hasattr(font, 'getbbox'):
                             _, ascent, _, descent = font.getmetrics() if hasattr(font, 'getmetrics') else (0,0,0,0)
                             text_h = (font.getbbox(text_label_on_image)[3] - font.getbbox(text_label_on_image)[1]) if ascent == 0 else ascent + descent
                        else: text_h = font_size_on_image * 1.2
                    else: text_w, text_h = draw.textsize(text_label_on_image, font=font)
                    text_bg_height = text_h + 4
                    text_y_position = y1 - text_bg_height
                    if text_y_position < 0: text_y_position = y1 + 2
                    text_bg_coords = [x1, text_y_position, x1 + text_w + 4, text_y_position + text_bg_height]
                except Exception as e_text_layout:
                    print(f"Warning: Error during text layout for drawing: {e_text_layout}. Using basic positioning.")
                    text_bg_coords = [x1, y1-20 if y1-20 > 0 else y1+2, x1+80, y1]; text_y_position = y1-20 if y1-20 > 0 else y1+2
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.rectangle(text_bg_coords, fill="red")
                draw.text((x1 + 2, text_y_position + 2), text_label_on_image, fill="white", font=font)
        buffered = io.BytesIO()
        annotated_img.save(buffered, format="PNG")
        img_str_base64 = base64.b64encode(buffered.getvalue()).decode()
        return JSONResponse(content={"predictions": predictions_data, "annotated_image_base64": img_str_base64, "filename": file.filename if file.filename else "uploaded_image"})
    except Exception as e:
        print(f"ERROR during prediction: {e}"); traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# --- Serve Frontend: Mount static directory and serve index.html ---
# This must be defined *after* your API endpoints if paths could conflict,
# or ensure paths are distinct. For root path serving index.html, it's often last.

# Mount the 'static' directory. Files in 'static' will be accessible e.g. /static/style.css
app.mount(f"/{STATIC_DIR}", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=FileResponse)
async def read_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        print(f"ERROR: index.html not found at {index_path}")
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)

# --- Uvicorn startup (for local dev) ---
if __name__ == "__main__":
    if MODEL_PATH == "best.pt" and not os.path.exists(MODEL_PATH):
        print("---------------------------------------------------------------------------")
        print(f"IMPORTANT: '{MODEL_PATH}' not found in the current directory ('{os.getcwd()}').")
        # ... (rest of the startup message)
        print("---------------------------------------------------------------------------")
    port = int(os.getenv("PORT", 8000))
    print(f"INFO: Starting Uvicorn on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)