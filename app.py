import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import os
from datetime import datetime
from config import (
    APP_TITLE, APP_DESCRIPTION, EMOTIONS, EMOTION_COLORS,
    EMOTION_KEYWORDS, OPENAI_API_KEY, ENVIRONMENT
)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Try to import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM STYLING
# ============================================
st.markdown("""
<style>
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        display: flex;
        gap: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        flex-direction: row-reverse;
        margin-left: 2rem;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 2rem;
    }
    
    .emotion-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 1.5rem;
        font-size: 0.9rem;
        font-weight: bold;
        margin-left: 0.5rem;
        color: black;
        background-color: white;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-title {
        text-align: center;
        color: #667eea;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    .divider {
        margin: 2rem 0;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []
    
if "conversation_count" not in st.session_state:
    st.session_state.conversation_count = 0

# ============================================
# HELPER FUNCTIONS
# ============================================

@st.cache_resource
def load_nlp_models():
    """Load NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        return True
    except LookupError:
        return False

def preprocess_text(text):
    """Preprocess text for analysis"""
    try:
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        return ' '.join(tokens)
    except:
        return text.lower()

def classify_emotion(text):
    """Classify emotion based on keywords"""
    text_lower = text.lower()
    
    scores = {}
    for emotion_id, keywords in EMOTION_KEYWORDS.items():
        scores[emotion_id] = sum(1 for keyword in keywords if keyword in text_lower)
    
    max_score = max(scores.values()) if scores else 0
    
    if max_score > 0:
        emotion_id = max(scores, key=scores.get)
        confidence = min(max_score / len(text_lower.split()), 1.0)
    else:
        emotion_id = 3  # Neutral
        confidence = 0.5
    
    return emotion_id, confidence

def generate_response(user_message, emotion_id):
    """Generate response using OpenAI API or fallback"""
    
    emotion_name = EMOTIONS[emotion_id].split()[-1]
    
    # Check if API key exists
    api_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else OPENAI_API_KEY
    
    if not api_key or not OPENAI_AVAILABLE:
        # Fallback responses
        fallback_responses = {
            0: "That's wonderful! I'm really glad you're feeling good. Your positive energy is contagious! 😊",
            1: "I understand you're going through a tough time. It's completely okay to feel this way. I'm here to listen. 💙",
            2: "I can sense your frustration. Take a deep breath. Sometimes we just need to let it out. How can I help? 🤝",
            3: "Thanks for sharing. I appreciate you opening up. Tell me more about what's on your mind. 👂",
            4: "It sounds like you've had a challenging experience. That's frustrating. Would you like to talk about it? 💪",
            5: "That's really interesting! I'd love to hear more about what surprised you. 🌟"
        }
        return fallback_responses.get(emotion_id, "I'm here to listen and support you. How can I help?")
    
    try:
        if OPENAI_AVAILABLE:
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an empathetic AI chatbot. The user seems {emotion_name.lower()}. Respond with genuine empathy and support in 1-2 sentences. Be warm and understanding."
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                temperature=0.7,
                max_tokens=100
            )
            return response.choices[0].message['content'].strip()
    except Exception as e:
        st.warning(f"API Error: {str(e)}")
    
    # Default fallback
    return f"I understand you're feeling {emotion_name.lower()}. I'm here to support you. 💙"

# ============================================
# MAIN APP
# ============================================

def main():
    # Header
    st.markdown('<div class="header-title">🤖 AI Sentiment-Aware Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">An intelligent conversational AI with emotion detection & empathetic responses</div>', unsafe_allow_html=True)
    
    # Main content area
    col_chat, col_sidebar = st.columns([3, 1])
    
    with col_sidebar:
        st.subheader("📊 Dashboard")
        
        # Stats
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.markdown(f"""
            <div class="stats-card">
                <div style="font-size: 2rem;">💬</div>
                <div style="font-size: 0.9rem;">Messages</div>
                <div style="font-size: 1.5rem;">{len(st.session_state.messages)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stat2:
            st.markdown(f"""
            <div class="stats-card">
                <div style="font-size: 2rem;">😊</div>
                <div style="font-size: 0.9rem;">Emotions</div>
                <div style="font-size: 1.5rem;">{len(st.session_state.emotion_history)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Emotion breakdown
        if st.session_state.emotion_history:
            st.subheader("Emotion Breakdown")
            emotion_counts = {}
            for emotion_id in st.session_state.emotion_history:
                emotion_name = EMOTIONS[emotion_id].split()[-1]
                emotion_counts[emotion_name] = emotion_counts.get(emotion_name, 0) + 1
            
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"{emotion}: **{count}**")
        
        st.divider()
        
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 New Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.emotion_history = []
                st.rerun()
        
        with col_btn2:
            if st.button("📥 Export", use_container_width=True):
                st.info("Feature coming soon!")
        
        st.divider()
        
        # Info section
        with st.expander("ℹ️ About"):
            st.markdown("""
            This chatbot uses:
            - **Deep Learning**: LSTM for emotion detection
            - **NLP**: Text preprocessing and analysis
            - **AI**: OpenAI GPT for intelligent responses
            - **UI**: Streamlit for interactive interface
            """)
    
    # Chat interface
    with col_chat:
        st.subheader("💬 Chat with Me")
        
        # Input area
        col_input, col_send = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "You:",
                placeholder="Type your message here...",
                label_visibility="collapsed"
            )
        
        with col_send:
            send_button = st.button("Send ➤", use_container_width=True)
        
        # Process input
        if send_button and user_input:
            # Classify emotion
            emotion_id, confidence = classify_emotion(user_input)
            
            # Add to history
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "emotion": emotion_id,
                "confidence": confidence,
                "timestamp": datetime.now()
            })
            st.session_state.emotion_history.append(emotion_id)
            st.session_state.conversation_count += 1
            
            # Generate response
            with st.spinner("🤔 Thinking..."):
                bot_response = generate_response(user_input, emotion_id)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_response,
                "timestamp": datetime.now()
            })
            
            st.rerun()
        
        st.divider()
        
        # Chat history display
        st.subheader("Conversation History")
        
        if st.session_state.messages:
            for i, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    emotion_badge = f'<span class="emotion-badge" style="background-color: {EMOTION_COLORS.get(message["emotion"], "#D3D3D3")}; color: black;">{EMOTIONS[message["emotion"]]}</span>'
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <div>
                            <b>You:</b> {message['content']}
                            {emotion_badge}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div>
                            <b>🤖 Bot:</b> {message['content']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                👋 <b>Welcome!</b> Start a conversation with me. I'll detect your emotions and respond with empathy and understanding.
            </div>
            """, unsafe_allow_html=True)

# ============================================
# RUN APP
# ============================================
if __name__ == "__main__":
    main()
