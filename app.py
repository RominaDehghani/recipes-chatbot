import streamlit as st
import os
import json
from dotenv import load_dotenv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# No googletrans, the app will be entirely in English.
# Removed TRANSLATOR_AVAILABLE and related logic.

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# Load .env (for local development, Streamlit Cloud uses Secrets)
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Gemini API configuration
def configure_genai():
    if not GENAI_AVAILABLE:
        st.error("Google Generative AI library is not installed.")
        return False
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not found. Please check your .env file or add it to Streamlit Secrets.")
        return False
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return True
    except Exception as e:
        st.error(f"Gemini API configuration error: {e}")
        return False

# --- Data & Retrieval Setup ---
@st.cache_data
def load_and_process_data():
    CSV_PATH = os.path.join(os.path.dirname(__file__), '13k-recipes.csv')
    if not os.path.exists(CSV_PATH):
        st.warning("13k-recipes.csv not found. Using sample data.")
        df_ = pd.DataFrame({
            'Title': ['Chicken Stir-fry', 'Lentil Soup', 'Pasta Salad', 'Spicy Shrimp Pasta', 'Vegetable Curry'],
            'Ingredients': ['Chicken, Bell Pepper, Onion, Tomato, Spices', 'Lentil, Onion, Carrot, Tomato Paste, Mint', 'Pasta, Mayonnaise, Peas, Carrot', 'Shrimp, Pasta, Garlic, Chili, Olive Oil', 'Mixed Vegetables, Coconut Milk, Curry Powder'],
            'Instructions': [
                'Stir-fry chicken, bell pepper and onion. Add chopped tomato and spices. Cook until chicken is done.',
                'Boil lentils with chopped onion and carrot. Add tomato paste and mint. Simmer until tender.',
                'Boil pasta. Mix with mayonnaise, peas, and shredded carrot. Chill and serve.',
                'Cook pasta. Sauté shrimp with garlic and chili. Combine with pasta and olive oil.',
                'Sauté mixed vegetables. Add coconut milk and curry powder. Simmer until vegetables are tender.'
            ],
            'Cleaned_Ingredients': [
                'chicken bell pepper onion tomato spices',
                'lentil onion carrot tomato paste mint',
                'pasta mayonnaise peas carrot',
                'shrimp pasta garlic chili olive oil',
                'mixed vegetables coconut milk curry powder'
            ]
        })
    else:
        df_ = pd.read_csv(CSV_PATH)
        expected_cols = ['Title', 'Ingredients', 'Instructions', 'Cleaned_Ingredients']
        for c in expected_cols:
            if c not in df_.columns:
                st.error(f"Column '{c}' not found in CSV. Please check your file.")
                return pd.DataFrame(), None
        df_ = df_[expected_cols].dropna().reset_index(drop=True)

    vectorizer_ = TfidfVectorizer()
    if not df_.empty:
        X_ = vectorizer_.fit_transform(df_['Cleaned_Ingredients'].astype(str))
    else:
        X_ = None
        st.warning("DataFrame is empty, TF-IDF vectorizer could not be created.")
    
    return df_, vectorizer_, X_

df, vectorizer, X = load_and_process_data()

def retrieve_recipes(user_input_en, top_n=3):
    """Returns top_n recipes (pandas DataFrame) similar to the user input."""
    if X is None or df.empty:
        return pd.DataFrame()
    
    input_vec = vectorizer.transform([user_input_en])
    similarity = cosine_similarity(input_vec, X).flatten()
    
    min_similarity = 0.05
    matching_indices = [i for i, score in enumerate(similarity) if score > min_similarity]
    
    matching_indices.sort(key=lambda i: similarity[i], reverse=True)
    
    top_indices = matching_indices[:top_n]
    return df.iloc[top_indices]

# --- Gemini API wrapper ---
def call_gemini(prompt, model_name='gemini-2.5-flash'): 
    """Calls Gemini and returns the text result. Returns a mock response if genai is unavailable."""
    if not GENAI_AVAILABLE or not GEMINI_API_KEY:
        mock_response = (
            "Hello! Access to the Gemini API is currently unavailable, so I'm giving you a sample response:\n\n"
            "<b>Recipe suggestions based on ingredients:</b>\n\n"
            "<h3>1. Simple Chicken Stir-fry</h3>"
            "This recipe allows you to make a great stir-fry with chicken, bell pepper, and onion.\n"
            "<b>Ingredients:</b>\n"
            "<ul><li>Chicken</li><li>Bell Pepper</li><li>Onion</li><li>Tomato</li><li>Spices</li></ul>\n"
            "<b>Instructions:</b>\n"
            "<ol><li>Cut the chicken into cubes.</li><li>Sauté with bell pepper and onion in olive oil.</li><li>Add salt, black pepper, and thyme.</li><li>Serve with rice.</li></ol>\n\n"
            "Enjoy your meal!"
        )
        return mock_response

    try:
        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content(prompt)
        if hasattr(response, 'text'):
            return response.text
        return str(response)
    except Exception as e:
        st.error(f"Sorry, an issue occurred while communicating with the Gemini API: {str(e)}")
        return f"Sorry, an issue occurred while communicating with the Gemini API: {str(e)}"

# --- Streamlit App ---
st.set_page_config(
    page_title="Delicious Recipe Assistant",
    page_icon="🍲",
    layout="centered"
)

# Custom CSS for a nicer look and chat alignment
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Open+Sans:wght@400;600&display=swap');

    :root {
        --primary-color: #a0522d; /* Sienna - Warm brown */
        --secondary-color: #f4a460; /* SandyBrown - Lighter warm brown */
        --background-color: #fdf5e6; /* OldLace - Creamy white */
        --chat-bg: #fffaf0; /* FloralWhite - Slightly warmer white */
        --text-color: #36454F; /* Charcoal */
        --border-color: #d2b48c; /* Tan */
        --user-bubble-bg: #e6a75a; /* A custom warm orange-brown */
        --bot-bubble-bg: #f5f5dc; /* Beige */
        --input-bg: #fff;
    }

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Open Sans', sans-serif;
        background-color: var(--background-color);
        color: var(--text-color);
    }

    [data-testid="stHeader"] {
        background-color: var(--background-color);
        visibility: hidden; /* Hide Streamlit's default header */
        height: 0px;
    }

    /* Main container styling */
    .stApp {
        background: var(--background-color);
        max-width: 800px;
        margin: auto;
        border-radius: 18px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
        overflow: hidden;
    }

    /* Chat header */
    .chat-header {
        padding: 25px;
        background: linear-gradient(135deg, var(--primary-color), #8b4513); /* Darker Sienna gradient */
        color: white;
        border-bottom: 2px solid #8b4513;
        text-align: center;
        font-family: 'Merriweather', serif;
        font-size: 1.8em;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-top-left-radius: 18px;
        border-top-right-radius: 18px;
        margin-bottom: 0px !important;
    }

    /* Chat messages container */
    [data-testid="stVerticalBlock"] > div:first-child {
        padding-top: 0px !important;
    }
    
    /* Override Streamlit's default chat message styling for alignment */
    [data-testid="chat-message-container"] {
        margin-bottom: 18px;
        display: flex;
        align-items: flex-start;
        padding: 0 10px;
        gap: 10px; /* Space between avatar and message */
    }

    /* Assistant (bot) message alignment (left) */
    [data-testid="chat-message-container"].stChatMessage:has(.stMarkdownContainer) {
        justify-content: flex-start;
    }
    
    /* User message alignment (right) */
    [data-testid="chat-message-container"].stChatMessage:has(.stMarkdownContainer) > div:first-child:empty { /* Targets the empty avatar div for user messages */
        order: 2; /* Move avatar to the right */
    }
    [data-testid="chat-message-container"].stChatMessage:has(.stMarkdownContainer) > div:nth-child(2) { /* Targets the message content div */
        order: 1; /* Move message content to the left */
    }
    [data-testid="chat-message-container"].stChatMessage.st-emotion-cache-1c7v0u5.user { /* Specific selector for user message bubble */
        flex-direction: row-reverse; /* Align user bubble to the right */
        justify-content: flex-end;
    }
    [data-testid="chat-message-container"].stChatMessage.st-emotion-cache-1c7v0u5.user .stMarkdownContainer {
        border-bottom-right-radius: 8px; /* Tail effect for user bubble */
        border-bottom-left-radius: 25px; /* Adjust if needed */
    }
    [data-testid="chat-message-container"].stChatMessage.st-emotion-cache-1c7v0u5.assistant .stMarkdownContainer {
        border-bottom-left-radius: 8px; /* Tail effect for bot bubble */
        border-bottom-right-radius: 25px; /* Adjust if needed */
    }


    /* Message content styling */
    .stChatMessage .stMarkdownContainer {
        padding: 14px 20px;
        border-radius: 25px;
        max-width: 75%;
        line-height: 1.6;
        font-size: 1em;
        word-wrap: break-word;
        white-space: pre-wrap; /* Preserve formatting from Gemini */
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .stChatMessage.user .stMarkdownContainer {
        background: var(--user-bubble-bg);
        color: white;
    }
    .stChatMessage.assistant .stMarkdownContainer {
        background: var(--bot-bubble-bg);
        color: var(--text-color);
        border: 1px solid var(--border-color);
    }

    /* Input area */
    [data-testid="stForm"] {
        padding: 18px 25px;
        border-top: 1px solid var(--border-color);
        background-color: var(--input-bg);
        border-bottom-left-radius: 18px;
        border-bottom-right-radius: 18px;
        margin-top: 0px !important;
    }

    .stTextInput > div > div > input {
        padding: 14px 20px;
        border: 1px solid var(--border-color);
        border-radius: 28px;
        outline: none;
        font-size: 1em;
        transition: border-color 0.3s, box-shadow 0.3s;
        background-color: var(--input-bg);
        color: var(--text-color);
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 4px rgba(160, 82, 45, 0.2);
    }

    .stButton > button {
        padding: 14px 28px;
        background: linear-gradient(90deg, var(--secondary-color), var(--primary-color));
        color: white;
        border: none;
        border-radius: 28px;
        cursor: pointer;
        font-size: 1.05em;
        font-weight: 600;
        transition: background 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.15);
        margin-left: 12px;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, var(--primary-color), #d2691e);
        transform: translateY(-2px);
        box-shadow: 0 7px 20px rgba(0,0,0,0.2);
    }
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }

    /* Custom styles for recipe output from Gemini (Streamlit will render HTML markdown) */
    .stMarkdown h3 {
        font-family: 'Merriweather', serif;
        color: var(--primary-color);
        margin-top: 1.5em;
        margin-bottom: 0.5em;
        font-size: 1.3em;
        font-weight: 700;
    }
    .stMarkdown p {
        margin-bottom: 0.8em;
    }
    .stMarkdown b {
        font-family: 'Merriweather', serif;
        color: var(--primary-color);
        display: block;
        margin-top: 1em;
        margin-bottom: 0.5em;
        font-size: 1.1em;
    }
    .stMarkdown ul, .stMarkdown ol {
        margin-left: 20px;
        margin-bottom: 1em;
        padding: 0;
        list-style-position: inside;
    }
    .stMarkdown ul li {
        list-style-type: '🍪 ';
        margin-bottom: 0.5em;
    }
    .stMarkdown ol li {
        margin-bottom: 0.5em;
    }
    /* Hide the Streamlit footer and header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-header">Delicious Recipe Assistant</div>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I am your Delicious Recipe Assistant. What ingredients would you like to start with to create wonders in the kitchen? 😊"})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Main chat input
if prompt := st.chat_input("Enter ingredients (e.g., chicken, mushrooms, cream)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Use a placeholder for the assistant's response to enable dynamic content
    # This ensures the spinner appears *before* processing and is replaced by the actual content
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Create an empty slot for the message
        with st.spinner("Thinking about recipes, I'll prepare them shortly..."):
            user_input_en = prompt.strip()

            if not user_input_en:
                response_content = 'Please specify the ingredients.'
                message_placeholder.markdown(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                st.stop() # Stop further execution for this run

            # --- Extract requested recipe count ---
            requested_num_recipes = 1 # Default to 1 recipe
            match = re.search(r'(?:give me|show me|find me|I need)\s*(\d+)\s*recipes?', user_input_en, re.IGNORECASE)
            if match:
                try:
                    num = int(match.group(1))
                    requested_num_recipes = max(1, min(num, 3)) # Enforce min 1, max 3
                    # Optionally remove the number request from the input passed to retrieval/Gemini
                    user_input_en = re.sub(r'(?:give me|show me|find me|I need)\s*\d+\s*recipes?', '', user_input_en, 1, re.IGNORECASE).strip()
                except ValueError:
                    pass # Keep default 1 if parsing fails

            # --- Intent Check ---
            intent_prompt = f"User's request: \"{user_input_en}\". Is this request related to a recipe or ingredient query? Answer only 'YES' or 'NO'."
            intent_response = call_gemini(intent_prompt)
            
            if "NO" in intent_response.upper():
                response_content = "Sorry, I can only help with recipes. Please ask a food-related question. 🧑‍🍳"
                message_placeholder.markdown(response_content) # Display directly
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                st.stop() # Stop further execution for this run

            # Retrieve recipes using the English input
            recipes = retrieve_recipes(user_input_en, top_n=requested_num_recipes)
            
            context = ""
            if recipes.empty:
                context = "No recipe directly matching the user's ingredients was found in the database."
            else:
                rows = []
                for _, row in recipes.iterrows():
                    rows.append(
                        f"<h3>{row['Title']}</h3>"
                        f"<b>Ingredients:</b>\n<ul><li>" + "</li><li>".join(row['Ingredients'].split(', ')) + "</li></ul>\n"
                        f"<b>Instructions:</b>\n<ol><li>" + "</li><li>".join(re.split(r'\d+\.\s*', str(row['Instructions']))[1:]) + "</li></ol>"
                    )
                context = "\n<hr>\n".join(rows)

            if recipes.empty:
                gemini_prompt = f"The user specified these ingredients: \"{user_input_en}\"\n\n" \
                                 f"Available recipes:\n{context}\n\n" \
                                 f"Since no recipe matching these ingredients was found in the database, please inform the user about this and offer a general recipe idea or an alternative suggestion that can be prepared with these ingredients. Respond in English, with a friendly and motivating tone. Please format your response using HTML tags (h3, b, ul, li, ol)."
            else:
                gemini_prompt = f"The user specified these ingredients: \"{user_input_en}\"\n\n" \
                                 f"Referring to the available recipes below, suggest the most suitable recipes for the user's ingredients. If you use a recipe as is, please maintain the given structure. If you generate a new idea, create a recipe in a similar format.\n\n" \
                                 f"Available Recipes:\n{context}\n\n" \
                                 f"Please respond in English, with a friendly and motivating tone. Format the recipes using HTML tags (h3 for title, b tags for 'Ingredients:' and 'Instructions:', ul/li for ingredient list, ol/li for steps). You can use a short and cheerful introductory sentence if needed. For example: 'Great choice! Let's see what we can make with these ingredients...'" \
                                 f"Suggest only {requested_num_recipes} recipe(s)."

            generated_response = call_gemini(gemini_prompt)
            message_placeholder.markdown(generated_response, unsafe_allow_html=True) # Replace spinner with actual content
            st.session_state.messages.append({"role": "assistant", "content": generated_response})