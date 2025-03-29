import os
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = "7567525625:AAHWELrjCj29vVpyv-ZVHyWdDUnHjX7CR3w"
API_URL = "https://fwrcrh04-8000.inc1.devtunnels.ms"  # Base API URL

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message with instructions"""
    await update.message.reply_text(
        '🤖 Welcome to DeepFake Detection Bot!\n\n'
        'I can analyze both images and videos for potential deepfake manipulation.\n\n'
        '📷 Send me an image (JPG/PNG) or\n'
        '🎥 Send me a video (MP4/MOV)\n\n'
        'Use /help for more information'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send help message"""
    await update.message.reply_text(
        '🆘 Help Menu 🆘\n\n'
        'How to use this bot:\n'
        '1. Send an image (JPG/PNG) or video (MP4/MOV)\n'
        '2. Wait for analysis (typically a few seconds)\n'
        '3. Receive detection results\n\n'
        'Commands:\n'
        '/start - Show welcome message\n'
        '/help - Show this help message\n\n'
        'The bot will automatically detect whether you sent an image or video.'
    )

async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle both image and video uploads"""
    # Determine media type
    if update.message.photo:
        # Get highest resolution photo
        file = update.message.photo[-1]
        media_type = "image"
        endpoint = "/image-detect"
    elif update.message.video:
        file = update.message.video
        media_type = "video"
        endpoint = "/video-detect"
    elif update.message.document:
        # Check if document is an image or video
        if update.message.document.mime_type.startswith('image/'):
            file = update.message.document
            media_type = "image"
            endpoint = "/image-detect"
        elif update.message.document.mime_type.startswith('video/'):
            file = update.message.document
            media_type = "video"
            endpoint = "/video-detect"
        else:
            await update.message.reply_text("Please send an image (JPG/PNG) or video (MP4/MOV) file")
            return
    else:
        await update.message.reply_text("Please send an image or video for analysis")
        return
    
    # Inform user that processing has started
    processing_msg = await update.message.reply_text(f"🔍 Analyzing {media_type} for deepfake manipulation...")
    
    try:
        # Get the file from Telegram servers
        file_id = file.file_id
        new_file = await context.bot.get_file(file_id)
        
        # Download the file temporarily
        file_ext = ".jpg" if media_type == "image" else ".mp4"
        file_path = f"temp_{file_id}{file_ext}"
        await new_file.download_to_drive(file_path)
        
        # Prepare API request
        api_url = f"{API_URL}{endpoint}"
        with open(file_path, 'rb') as f:
            files = {'file': f}
            params = {'num_frames': 15} if media_type == "video" else None
            response = requests.post(api_url, files=files, params=params)
        
        # Delete the temporary file
        os.remove(file_path)
        
        if response.status_code == 200:
            result = response.json()
            confidence_percent = round(result['confidence'] * 100, 2)
            
            if result['is_fake']:
                message = (
                    f"🚨 DEEPFAKE DETECTED 🚨\n\n"
                    f"Media type: {media_type.upper()}\n"
                    f"Confidence: {confidence_percent}%\n"
                )
                if media_type == "video":
                    message += f"Processed frames: {result['processed_frames']}\n"
                message += (
                    f"Filename: {result['filename']}\n\n"
                    f"This {media_type} appears to be manipulated."
                )
            else:
                message = (
                    f"✅ Authentic Content ✅\n\n"
                    f"Media type: {media_type.upper()}\n"
                    f"Confidence: {confidence_percent}%\n"
                )
                if media_type == "video":
                    message += f"Processed frames: {result['processed_frames']}\n"
                message += (
                    f"Filename: {result['filename']}\n\n"
                    f"This {media_type} appears to be genuine."
                )
            
            # Edit the original processing message with results
            await processing_msg.edit_text(message)
        else:
            await processing_msg.edit_text("❌ Error analyzing media. Please try again later.")
            logger.error(f"API Error: {response.status_code} - {response.text}")
    
    except Exception as e:
        logger.error(f"Error processing {media_type}: {str(e)}")
        await processing_msg.edit_text(f"❌ An error occurred while processing the {media_type}. Please try again.")

def main() -> None:
    """Start the bot."""
    # Create the Application
    application = Application.builder().token(TOKEN).build()

    # Register command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # Register media handlers
    application.add_handler(MessageHandler(
        filters.PHOTO | filters.VIDEO | 
        (filters.Document.IMAGE | filters.Document.VIDEO),
        handle_media
    ))

    # Start the Bot
    application.run_polling()

if __name__ == '__main__':
    main()
