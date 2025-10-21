import streamlit as st
import os
import json
from dotenv import load_dotenv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Import googletrans for translation
try:
    from googletrans import Translator
    TRANSLATOR_AVAILABLE = True
    translator = Translator()
except Exception as e:
    TRANSLATOR_AVAILABLE = False
    st.warning(f"Uyarı: googletrans kütüphanesi yüklenemedi: {e}. Çeviri katmanı devre dışı.")

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# Load .env (Streamlit Cloud'da secret olarak yüklenecek)
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Gemini API yapılandırması
def configure_genai():
    if not GENAI_AVAILABLE:
        st.error("Google Generative AI kütüphanesi yüklü değil.")
        return False
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY bulunamadı. Lütfen .env dosyasını kontrol edin veya Streamlit Secrets'a ekleyin.")
        return False
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return True
    except Exception as e:
        st.error(f"Gemini API yapılandırma hatası: {e}")
        return False

# --- Data & Retrieval Setup ---
@st.cache_data
def load_and_process_data():
    CSV_PATH = os.path.join(os.path.dirname(__file__), '13k-recipes.csv')
    if not os.path.exists(CSV_PATH):
        st.warning("13k-recipes.csv dosyası bulunamadı. Örnek veri kullanılıyor.")
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
                st.error(f"CSV dosyasında '{c}' sütunu bulunamadı. Lütfen dosyanızı kontrol edin.")
                return pd.DataFrame(), None # Hata durumunda boş DataFrame ve None dön
        df_ = df_[expected_cols].dropna().reset_index(drop=True)

    vectorizer_ = TfidfVectorizer()
    if not df_.empty:
        X_ = vectorizer_.fit_transform(df_['Cleaned_Ingredients'].astype(str))
    else:
        X_ = None
        st.warning("DataFrame boş, TF-IDF vektörleyicisi oluşturulamadı.")
    
    return df_, vectorizer_, X_

df, vectorizer, X = load_and_process_data()

def retrieve_recipes(user_input_en, top_n=3):
    """Kullanıcı girdisine benzer top_n tarifi (pandas DataFrame) döndürür."""
    if X is None or df.empty:
        return pd.DataFrame() # Veri yoksa boş DataFrame dön
    
    input_vec = vectorizer.transform([user_input_en])
    similarity = cosine_similarity(input_vec, X).flatten()
    
    min_similarity = 0.05  # Gerekirse benzerlik eşiğini ayarlayabilirsiniz
    matching_indices = [i for i, score in enumerate(similarity) if score > min_similarity]
    
    matching_indices.sort(key=lambda i: similarity[i], reverse=True)
    
    top_indices = matching_indices[:top_n]
    return df.iloc[top_indices]

# --- Gemini API sarmalayıcı ---
def call_gemini(prompt, model_name='gemini-2.5-flash'): 
    """Gemini'yi çağırır ve metin sonucu döndürür. genai mevcut değilse, örnek (mock) yanıt döner."""
    if not GENAI_AVAILABLE or not GEMINI_API_KEY:
        mock_response = (
            "Merhaba! Gemini API'sine erişim şu anda mümkün değil, bu yüzden size örnek bir yanıt veriyorum:\n\n"
            "<b>Malzemelerle ilgili tarif önerileri:</b>\n\n"
            "<h3>1. Basit Tavuk Sote</h3>"
            "Bu tarif, elinizdeki tavuk, biber ve soğanla harika bir sote yapmanızı sağlar.\n"
            "<b>Malzemeler:</b>\n"
            "<ul><li>Tavuk</li><li>Biber</li><li>Soğan</li><li>Domates</li><li>Baharatlar</li></ul>\n"
            "<b>Yapılışı:</b>\n"
            "<ol><li>Tavukları küp küp doğrayın.</li><li>Biber ve soğanla birlikte zeytinyağında soteleyin.</li><li>Tuz, karabiber ve kekik ekleyin.</li><li>Yanında pilavla servis edebilirsiniz.</li></ol>\n\n"
            "Afiyet olsun!"
        )
        return mock_response

    try:
        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content(prompt)
        if hasattr(response, 'text'):
            return response.text
        return str(response)
    except Exception as e:
        st.error(f"Üzgünüm, Gemini API ile iletişim kurarken bir sorun oluştu: {str(e)}")
        return f"Üzgünüm, Gemini API ile iletişim kurarken bir sorun oluştu: {str(e)}"

# --- Streamlit Uygulaması ---
st.set_page_config(
    page_title="Lezzetli Tarif Asistanı",
    page_icon="🍲",
    layout="centered"
)

# Custom CSS for a nicer look
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
        margin-bottom: 0px !important; /* Remove space below header */
    }

    /* Chat messages container */
    [data-testid="stVerticalBlock"] > div:first-child { /* Targets the main content block */
        padding-top: 0px !important;
    }
    .stChatMessage {
        margin-bottom: 18px;
        display: flex;
        align-items: flex-start;
        padding: 0 10px; /* Add some padding to messages */
    }
    .stChatMessage.user {
        justify-content: flex-end;
    }
    .stChatMessage.assistant {
        justify-content: flex-start;
    }

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
        border-bottom-right-radius: 8px; /* Tail effect */
    }
    .stChatMessage.assistant .stMarkdownContainer {
        background: var(--bot-bubble-bg);
        color: var(--text-color);
        border: 1px solid var(--border-color);
        border-bottom-left-radius: 8px; /* Tail effect */
    }

    /* Input area */
    [data-testid="stForm"] {
        padding: 18px 25px;
        border-top: 1px solid var(--border-color);
        background-color: var(--input-bg);
        border-bottom-left-radius: 18px;
        border-bottom-right-radius: 18px;
        margin-top: 0px !important; /* Remove space above input */
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
        background: linear-gradient(90deg, var(--primary-color), #d2691e); /* Chocolate-like hover */
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
    /* Hide the Streamlit footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="chat-header">Lezzetli Tarif Asistanı</div>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Merhaba! Ben Lezzetli Tarif Asistanınızım. Mutfakta harikalar yaratmak için hangi malzemelerle yola çıkmak istersiniz? 😊"})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Main chat input
if prompt := st.chat_input("Malzemeleri yazın (örn: tavuk, mantar, krema)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Tarifleri düşünüyorum, birazdan hazırlarım..."):
            user_input_tr = prompt.strip()
            top_n_for_retrieval = 1 # Default to 1 recipe for simplicity, can be adjusted

            if not user_input_tr:
                st.markdown('Lütfen malzemeleri belirtin.')
                st.stop()

            # --- Intent Check ---
            intent_prompt = f"Kullanıcının isteği: \"{user_input_tr}\". Bu istek yemek tarifi veya malzeme sorgusu ile ilgili mi? Sadece 'EVET' veya 'HAYIR' olarak yanıt ver."
            intent_response = call_gemini(intent_prompt)
            
            if "HAYIR" in intent_response.upper():
                st.markdown("Üzgünüm, ben sadece yemek tarifleri konusunda yardımcı olabiliyorum. Lütfen yemekle ilgili bir soru sorun. 🧑‍🍳")
                st.session_state.messages.append({"role": "assistant", "content": "Üzgünüm, ben sadece yemek tarifleri konusunda yardımcı olabiliyorum. Lütfen yemekle ilgili bir soru sorun. 🧑‍🍳"})
                st.stop()

            user_input_en = user_input_tr
            if TRANSLATOR_AVAILABLE:
                try:
                    translated_text = translator.translate(user_input_tr, src='tr', dest='en').text
                    user_input_en = translated_text
                    st.sidebar.write(f"[TRANSLATION] TR: '{user_input_tr}' -> EN: '{user_input_en}'") # For debugging
                except Exception as e:
                    st.sidebar.warning(f"[TRANSLATION ERROR] Could not translate: {e}. Using original input for retrieval.")
            else:
                st.sidebar.warning("[TRANSLATION] Translator not available. Proceeding with original (Turkish) input for retrieval.")
            
            recipes = retrieve_recipes(user_input_en, top_n=top_n_for_retrieval)
            
            context = ""
            if recipes.empty:
                context = "Veritabanında kullanıcının belirttiği malzemelerle doğrudan eşleşen bir tarif bulunamadı."
            else:
                rows = []
                for _, row in recipes.iterrows():
                    # HTML etiketlerini doğrudan Gemini'nin çıktısında olmasını istediğimiz formatta ayarlıyoruz.
                    rows.append(
                        f"<h3>{row['Title']}</h3>"
                        f"<b>Malzemeler:</b>\n<ul><li>" + "</li><li>".join(row['Ingredients'].split(', ')) + "</li></ul>\n"
                        f"<b>Yapılışı:</b>\n<ol><li>" + "</li><li>".join(re.split(r'\d+\.\s*', str(row['Instructions']))[1:]) + "</li></ol>"
                    )
                context = "\n<hr>\n".join(rows)

            if recipes.empty:
                gemini_prompt = f"Kullanıcı şu malzemeleri belirtti: \"{user_input_tr}\"\n\n" \
                                 f"Mevcut tarifler:\n{context}\n\n" \
                                 f"Veritabanında bu malzemelerle eşleşen bir tarif bulunamadığı için, lütfen kullanıcıya bu durumu bildir ve bu malzemelerle hazırlanabilecek genel bir tarif fikri veya alternatif bir öneri sun. Yanıtını Türkçe, samimi ve motive edici bir dille ver. Lütfen HTML etiketleri (h3, b, ul, li, ol) kullanarak formatla."
            else:
                gemini_prompt = f"Kullanıcı şu malzemeleri belirtti: \"{user_input_tr}\"\n\n" \
                                 f"Aşağıdaki mevcut tarifleri referans alarak kullanıcının malzemelerine en uygun tarifleri öner. Eğer tarifi aynen kullanırsan, lütfen verilen yapıyı koru. Eğer yeni bir fikir üretirsen, benzer formatta bir tarif oluştur.\n\n" \
                                 f"Mevcut Tarifler:\n{context}\n\n" \
                                 f"Lütfen yanıtını Türkçe, samimi ve motive edici bir dille ver. Tarifleri HTML etiketleri (h3 başlık için, b etiketleri 'Malzemeler:' ve 'Yapılışı:' için, ul/li malzeme listesi için, ol/li yapılış adımları için) kullanarak formatla. Gerekirse kısa ve sevimli bir giriş cümlesi kullanabilirsin. Örneğin: 'Harika bir seçim! Bu malzemelerle neler yapabiliriz bakalım...'" \
                                 f"Sadece {top_n_for_retrieval} adet tarif öner."

            generated_response = call_gemini(gemini_prompt)
            st.markdown(generated_response, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": generated_response})