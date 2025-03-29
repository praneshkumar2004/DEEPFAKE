# DeepFake Detection System

## Project Introduction
The DeepFake Detection System is a cutting-edge solution that combines FastAPI for backend processing and a Telegram Bot for user interaction. The system analyzes uploaded images and videos to detect potential deepfake manipulations using the GenConViT deep learning model.

---

## Key Features
- ✅ **Dual Detection Support:** Works with both images (JPG, PNG) and videos (MP4, MOV, AVI, WMV).
- ✅ Telegram Bot Integration: Users can upload media directly via the bot @DeepFakeIntelli_bot using PORT FORWARDING to establish the connection.
- ✅ **FastAPI Backend:** Handles deepfake detection via REST API endpoints (`/video-detect`, `/image-detect`).
- ✅ **QR Code Access:** Easily share the bot via a scannable QR code.
- ✅ **Real-Time Results:** Provides confidence scores and detection outcomes.

---
  ## QR for Telegram Bot
  <img src="https://github.com/user-attachments/assets/16f81f0f-3277-4f2c-8622-ddf0c4d20009" alt="QR Code" width="300" height="300">

  
  ## System Architecture Diagram
  ![System Architecture Diagram](https://github.com/user-attachments/assets/f7a75a15-3c74-4749-bdb2-8171f306be93)


## Demo Video
https://github.com/user-attachments/assets/4c407d64-fdd9-45a2-8036-8b4594f6e45a


## How It Works
1. **User Uploads Media:** Users upload images or videos via the Telegram bot or API.
2. **Backend Processing:** Media is processed using the GenConViT model.
3. **Results Returned:**
   - Prediction: **FAKE** or **REAL**
   - Confidence percentage
   - Processed frames (for videos)

---

## Technologies Used
- 🔹 **Python:** Backend logic and bot handling.
- 🔹 **FastAPI:** REST API for deepfake detection.
- 🔹 **GenConViT Model:** Deep learning model for detecting deepfakes.
- 🔹 **Telegram Bot API:** For user interactions.
- 🔹 **OpenCV & PIL:** Image and video processing.
- 🔹 **QR Code Generation:** For easy bot sharing.

---

## Use Cases
- 🛡️ **Social Media Verification:** Detect manipulated profile pictures and videos.
- 📰 **News Fact-Checking:** Identify AI-generated fake news media.
- 🔒 **Security Applications:** Prevent deepfake-based fraud.

---

## Next Steps
- 🚀 **Cloud Deployment:** Deploy the API on AWS or Azure.
- 📱 **Bot Enhancement:** Add more interactive features.
- 🔍 **Model Improvement:** Increase accuracy with more training data.

This project aims to provide a user-friendly, automated solution for detecting AI-generated deepfakes, helping to combat misinformation in digital media. 🚀

