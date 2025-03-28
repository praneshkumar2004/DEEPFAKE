# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse
# import os
# import tempfile
# import shutil
# import torch
# from huggingface_hub import hf_hub_download
# import cv2
# import time
# import sys
# from datetime import datetime

# # Add custom path for model imports
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # Import prediction functions
# from model.pred_func import (
#     load_genconvit,
#     df_face,
#     pred_vid,
#     real_or_fake
# )
# from model.config import load_config

# app = FastAPI(
#     title="GenConViT Deepfake Detection API",
#     description="API for detecting deepfake videos using GenConViT model",
#     version="1.0"
# )

# # Global variables for model
# MODEL = None
# CONFIG = None
# MODEL_LOADED = False

# def load_model():
#     """Load the model weights from Hugging Face Hub"""
#     global MODEL, CONFIG, MODEL_LOADED
    
#     if not MODEL_LOADED:
#         try:
#             config = load_config()
            
#             # Create weights directory if not exists
#             os.makedirs("weight", exist_ok=True)
            
#             # Download model weights
#             ed_path = hf_hub_download(
#                 repo_id="Deressa/GenConViT",
#                 filename="genconvit_ed_inference.pth",
#             )
#             vae_path = hf_hub_download(
#                 repo_id="Deressa/GenConViT",
#                 filename="genconvit_vae_inference.pth",
#             )
            
#             shutil.copy(ed_path, "weight/genconvit_ed_inference.pth")
#             shutil.copy(vae_path, "weight/genconvit_vae_inference.pth")
            
#             # Load model
#             MODEL = load_genconvit(
#                 config,
#                 "genconvit",
#                 "genconvit_ed_inference",
#                 "genconvit_vae_inference",
#                 fp16=False
#             )
#             CONFIG = config
#             MODEL_LOADED = True
            
#         except Exception as e:
#             raise RuntimeError(f"Failed to load model: {str(e)}")

# @app.on_event("startup")
# async def startup_event():
#     """Load model on startup"""
#     try:
#         load_model()
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         raise

# def is_video(file_path: str) -> bool:
#     """Check if file is a valid video"""
#     try:
#         cap = cv2.VideoCapture(file_path)
#         if not cap.isOpened():
#             return False
#         ret, frame = cap.read()
#         cap.release()
#         return ret
#     except:
#         return False

# def process_video(video_path: str, num_frames: int = 15) -> dict:
#     """Process video and return prediction"""
#     try:
#         # Extract faces from video
#         df = df_face(video_path, num_frames, "genconvit")
        
#         if len(df) < 1:
#             return {
#                 "status": "error",
#                 "message": "No faces detected in video",
#                 "prediction": None,
#                 "confidence": None
#             }
        
#         # Get prediction
#         y, y_val = pred_vid(df, MODEL)
#         prediction = real_or_fake(y)
#         confidence = float(y_val)
        
#         return {
#             "status": "success",
#             "prediction": prediction,
#             "confidence": confidence,
#             "is_fake": prediction == "FAKE",
#             "processed_frames": len(df)
#         }
        
#     except Exception as e:
#         return {
#             "status": "error",
#             "message": str(e),
#             "prediction": None,
#             "confidence": None
#         }

# @app.post("/detect")
# async def detect_deepfake(
#     file: UploadFile = File(..., description="Video file to analyze"),
#     num_frames: int = 15
# ):
#     """Endpoint for deepfake detection"""
#     if not MODEL_LOADED:
#         raise HTTPException(status_code=503, detail="Model not loaded")
    
#     # Validate file type
#     if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.wmv')):
#         raise HTTPException(status_code=400, detail="Invalid file format. Supported formats: MP4, AVI, MOV, WMV")
    
#     # Save uploaded file temporarily
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
#         tmp_file.write(await file.read())
#         tmp_path = tmp_file.name
    
#     try:
#         # Validate video
#         if not is_video(tmp_path):
#             raise HTTPException(status_code=400, detail="Invalid video file")
        
#         # Process video
#         result = process_video(tmp_path, num_frames)
        
#         if result["status"] == "error":
#             raise HTTPException(status_code=400, detail=result["message"])
        
#         # Prepare response
#         response = {
#             "filename": file.filename,
#             "detection_result": result["prediction"],
#             "confidence": result["confidence"],
#             "is_fake": result["is_fake"],
#             "processed_frames": result["processed_frames"],
#             "timestamp": datetime.now().isoformat()
#         }
        
#         return JSONResponse(content=response)
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
#     finally:
#         # Clean up temporary file
#         if os.path.exists(tmp_path):
#             os.unlink(tmp_path)

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "OK" if MODEL_LOADED else "MODEL_NOT_LOADED",
#         "timestamp": datetime.now().isoformat()
#     }



from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
import shutil
import torch
from huggingface_hub import hf_hub_download
import cv2
import sys
from datetime import datetime
from PIL import Image
import numpy as np

# Add custom path for model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import prediction functions
from model.pred_func import (
    load_genconvit,
    df_face,
    pred_vid,
    real_or_fake
)
from model.config import load_config

app = FastAPI(
    title="GenConViT Deepfake Detection API",
    description="API for detecting deepfake images and videos using GenConViT model",
    version="1.0"
)

# Global variables for model
MODEL = None
CONFIG = None
MODEL_LOADED = False

def load_model():
    """Load the model weights from Hugging Face Hub"""
    global MODEL, CONFIG, MODEL_LOADED
    
    if not MODEL_LOADED:
        try:
            config = load_config()
            
            # Create weights directory if not exists
            os.makedirs("weight", exist_ok=True)
            
            # Download model weights
            ed_path = hf_hub_download(
                repo_id="Deressa/GenConViT",
                filename="genconvit_ed_inference.pth",
            )
            vae_path = hf_hub_download(
                repo_id="Deressa/GenConViT",
                filename="genconvit_vae_inference.pth",
            )
            
            shutil.copy(ed_path, "weight/genconvit_ed_inference.pth")
            shutil.copy(vae_path, "weight/genconvit_vae_inference.pth")
            
            # Load model
            MODEL = load_genconvit(
                config,
                "genconvit",
                "genconvit_ed_inference",
                "genconvit_vae_inference",
                fp16=False
            )
            CONFIG = config
            MODEL_LOADED = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def is_video(file_path: str) -> bool:
    """Check if file is a valid video"""
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return ret
    except:
        return False

def is_image(file_path: str) -> bool:
    """Check if file is a valid image"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False

def convert_image_to_video(image_path: str) -> str:
    """Convert image to a single-frame video for processing"""
    try:
        # Read the image
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # Create temporary video path
        video_path = f"{image_path}_temp.mp4"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 1, (width, height))
        
        # Write the frame
        out.write(img)
        out.release()
        
        return video_path
    except Exception as e:
        raise RuntimeError(f"Failed to convert image to video: {str(e)}")

def process_video(video_path: str, num_frames: int = 15) -> dict:
    """Process video and return prediction"""
    try:
        # Extract faces from video
        df = df_face(video_path, num_frames, "genconvit")
        
        if len(df) < 1:
            return {
                "status": "error",
                "message": "No faces detected",
                "prediction": None,
                "confidence": None
            }
        
        # Get prediction
        y, y_val = pred_vid(df, MODEL)
        prediction = real_or_fake(y)
        confidence = float(y_val)
        
        return {
            "status": "success",
            "prediction": prediction,
            "confidence": confidence,
            "is_fake": prediction == "FAKE",
            "processed_frames": len(df)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "prediction": None,
            "confidence": None
        }

@app.post("/video-detect")
async def detect_video_deepfake(
    file: UploadFile = File(..., description="Video file to analyze"),
    num_frames: int = 15
):
    """Endpoint for video deepfake detection"""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.wmv')):
        raise HTTPException(status_code=400, detail="Invalid file format. Supported formats: MP4, AVI, MOV, WMV")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(await file.read())
        tmp_path = tmp_file.name
    
    try:
        # Validate video
        if not is_video(tmp_path):
            raise HTTPException(status_code=400, detail="Invalid video file")
        
        # Process video
        result = process_video(tmp_path, num_frames)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        # Prepare response
        response = {
            "filename": file.filename,
            "detection_result": result["prediction"],
            "confidence": result["confidence"],
            "is_fake": result["is_fake"],
            "processed_frames": result["processed_frames"],
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/image-detect")
async def detect_image_deepfake(
    file: UploadFile = File(..., description="Image file to analyze")
):
    """Endpoint for image deepfake detection (converts to single-frame video)"""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        raise HTTPException(status_code=400, detail="Invalid file format. Supported formats: JPG, JPEG, PNG, WEBP")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(await file.read())
        tmp_path = tmp_file.name
    
    try:
        # Validate image
        if not is_image(tmp_path):
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert image to single-frame video
        video_path = convert_image_to_video(tmp_path)
        
        try:
            # Process as single-frame video
            result = process_video(video_path, num_frames=1)
            
            if result["status"] == "error":
                raise HTTPException(status_code=400, detail=result["message"])
            
            # Prepare response
            response = {
                "filename": file.filename,
                "detection_result": result["prediction"],
                "confidence": result["confidence"],
                "is_fake": result["is_fake"],
                "processed_frames": 1,  # Single frame for images
                "timestamp": datetime.now().isoformat()
            }
            
            return JSONResponse(content=response)
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "OK" if MODEL_LOADED else "MODEL_NOT_LOADED",
        "timestamp": datetime.now().isoformat()
    }