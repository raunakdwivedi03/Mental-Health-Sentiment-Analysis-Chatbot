import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================
# OpenAI API Configuration
# ============================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_TEMPERATURE = 0.7
OPENAI_MAX_TOKENS = 100

# ============================================
# Application Configuration
# ============================================
APP_TITLE = "AI Sentiment-Aware Chatbot"
APP_DESCRIPTION = "An intelligent conversational AI with emotion detection"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ============================================
# Streamlit Configuration
# ============================================
STREAMLIT_THEME = os.getenv("STREAMLIT_THEME", "light")
STREAMLIT_LAYOUT = "wide"

# ============================================
# Model Settings
# ============================================
EMOTION_CONFIDENCE_THRESHOLD = float(os.getenv("EMOTION_CONFIDENCE_THRESHOLD", "0.5"))
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "50"))

# Emotion mappings
EMOTIONS = {
    0: "😊 Happy",
    1: "😢 Sad",
    2: "😠 Angry",
    3: "😐 Neutral",
    4: "😤 Frustrated",
    5: "😲 Surprised"
}

EMOTION_COLORS = {
    0: "#FFD700",   # Happy - Gold
    1: "#87CEEB",   # Sad - Sky blue
    2: "#FF6B6B",   # Angry - Red
    3: "#D3D3D3",   # Neutral - Light gray
    4: "#FF8C00",   # Frustrated - Orange
    5: "#9370DB"    # Surprised - Purple
}

# ============================================
# API Configuration
# ============================================
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# ============================================
# Logging Configuration
# ============================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ============================================
# NLP Configuration
# ============================================
STOP_WORDS_LANGUAGE = "english"
TEXT_PREPROCESSING_ENABLED = True

# ============================================
# Emotion Keywords for Classification
# ============================================
EMOTION_KEYWORDS = {
    0: ["happy", "great", "wonderful", "excellent", "awesome", "good", "love", "joy", "excited"],
    1: ["sad", "depressed", "upset", "unhappy", "terrible", "bad", "hate", "miserable", "lonely"],
    2: ["angry", "furious", "mad", "irritated", "annoyed", "rage", "hostile"],
    3: ["okay", "fine", "normal", "usual", "regular", "neutral", "alright"],
    4: ["frustrated", "annoyed", "tired", "exhausted", "fed up", "irritated"],
    5: ["surprised", "shocked", "amazed", "wow", "unexpected", "wonder"]
}

# ============================================
# Validation
# ============================================
if not OPENAI_API_KEY and ENVIRONMENT == "production":
    raise ValueError("OPENAI_API_KEY is required in production mode")

# ============================================
# Feature Flags
# ============================================
ENABLE_EMOTION_TRACKING = True
ENABLE_CONVERSATION_HISTORY = True
ENABLE_SENTIMENT_ANALYSIS = True
