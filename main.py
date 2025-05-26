import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import uvicorn
import base64
import torch
import traceback

MODEL_PATH = "best.pt"
CLASS_NAMES_CONFIG = ['text', 'metal']

app = FastAPI(title="Metal-Text Detection API")

origins = ["*"]  # Allows all origins. For production, restrict to your frontend's domain.
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

model = None
model_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"INFO: Attempting to load model on device: {model_device}")

try:
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Model file not found at '{os.getcwd()}/{MODEL_PATH}'.")
    else:
        model = YOLO(MODEL_PATH)
        model.to(model_device)
        print(f"INFO: Successfully loaded YOLOv8 model from {MODEL_PATH} to {model_device}")
        print(f"INFO: Model class names (from model file): {model.names}")
        # Verification against configured class names
        if model.names and isinstance(model.names, dict):
            model_class_list = [model.names[i] for i in sorted(model.names.keys())]
            if CLASS_NAMES_CONFIG != model_class_list:
                print("WARNING: Configured CLASS_NAMES_CONFIG does not match model's internal class names/order!")
                print(f"  Configured: {CLASS_NAMES_CONFIG}")
                print(f"  Model has: {model_class_list}")
        elif not model.names:
            print("WARNING: Model does not have internal class names. Using CLASS_NAMES_CONFIG.")
except Exception as e:
    print(f"CRITICAL ERROR: Error loading YOLOv8 model: {e}")
    traceback.print_exc()


@app.on_event("startup")
async def startup_event():
    if model is None:
        print("WARNING: YOLO Model was not loaded. The /predict endpoint will not work.")
    else:
        print("INFO: YOLO Model loaded. API is ready.")


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Metal-Text API. POST to /predict to detect objects."}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to load. API unavailable.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))
        original_mode = img.mode

        # Ensure image is in a drawable format (RGB or RGBA)
        if original_mode not in ['RGB', 'RGBA']:
            if original_mode in ['P', 'LA', 'L']:  # Palette, Grayscale with Alpha, Grayscale
                img = img.convert('RGBA')  # Convert to RGBA to handle transparency & allow color drawing
            else:  # Other modes
                img = img.convert('RGB')  # Fallback to RGB

        results = model.predict(source=img, device=model_device)

        predictions_data = []
        # Ensure annotated_img is RGBA for drawing and PNG saving
        annotated_img = img.copy()
        if annotated_img.mode != 'RGBA':
            annotated_img = annotated_img.convert('RGBA')
        draw = ImageDraw.Draw(annotated_img)

        if results and len(results) > 0:
            result = results[0]
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = round(box.conf[0].item(), 2)
                cls_id = int(box.cls[0].item())

                # Use class name from the model's internal names if available
                class_name_to_display = model.names.get(cls_id, f"ID_{cls_id}") if model.names else CLASS_NAMES_CONFIG[
                    cls_id] if 0 <= cls_id < len(CLASS_NAMES_CONFIG) else f"ID_{cls_id}"

                predictions_data.append({
                    "class_id": cls_id,
                    "class_name": class_name_to_display,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })

                text_fill_color = "white"
                if class_name_to_display == "text":
                    outline_color = "red"
                    text_bg_color = "red"
                else:
                    outline_color = "orange"
                    text_bg_color = "orange"

                try:
                    font = ImageFont.truetype("arial.ttf", 25)
                except IOError:  # If arial.ttf is not found (e.g. on Azure App Service)
                    font = ImageFont.load_default()

                text_label = f"{class_name_to_display} {int(conf * 100)}%"

                try:  # Pillow 9.2.0+ for textbbox with anchor
                    text_anchor_bbox = draw.textbbox((x1, y1), text_label, font=font, anchor="ls")  # ls = left-baseline
                    text_x = x1
                    text_y = y1 - (text_anchor_bbox[3] - text_anchor_bbox[1]) - 4  # Position above box
                    if text_y < 0: text_y = y1 + 2  # If too high, put below
                    final_text_bbox = draw.textbbox((text_x, text_y), text_label, font=font)
                except TypeError:  # Fallback for older Pillow (textsize, no anchor in textbbox)
                    # Estimate text size
                    text_w, text_h = draw.textsize(text_label, font=font)
                    text_y = y1 - text_h - 2 if y1 - text_h - 2 > 0 else y1 + 2
                    final_text_bbox = (x1, text_y, x1 + text_w, text_y + text_h)

                draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=3)
                draw.rectangle(
                    [final_text_bbox[0] - 1, final_text_bbox[1] - 1, final_text_bbox[2] + 1, final_text_bbox[3] + 1],
                    fill=text_bg_color)
                draw.text((final_text_bbox[0], final_text_bbox[1]), text_label, fill=text_fill_color, font=font)

        buffered = io.BytesIO()
        annotated_img.save(buffered, format="PNG")
        img_str_base64 = base64.b64encode(buffered.getvalue()).decode()

        return JSONResponse(content={
            "predictions": predictions_data,
            "annotated_image_base64": img_str_base64,
            "filename": file.filename if file.filename else "uploaded_image"
        })

    except Exception as e:
        print(f"ERROR during prediction: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use PORT from env for Azure, default to 8000 local
    print(f"INFO: Starting Uvicorn on 0.0.0.0:{port}")
    # Note: For local dev, `uvicorn main:app --reload` is often preferred in terminal
    uvicorn.run(app, host="0.0.0.0", port=port)