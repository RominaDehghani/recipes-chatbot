import os
import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
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
    print(f"Uyarı: googletrans kütüphanesi yüklenemedi: {e}. Çeviri katmanı devre dışı.")


try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# Load .env
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# --- Diagnostics helpers ---
def _mask_key(value):
    try:
        if not value:
            return 'None'
        value = str(value)
        if len(value) <= 8:
            return '*' * len(value)
        return f"{value[:4]}...{value[-4:]}"
    except Exception:
        return 'unavailable'


app = Flask(__name__)
CORS(app)

# Gemini API yapılandırması
def configure_genai():
    if not GENAI_AVAILABLE:
        
        return False
    if not GEMINI_API_KEY:
        
        return False
    try:
        genai.configure(api_key=GEMINI_API_KEY)
       
    except Exception as e:
        
        return False
    return True

# --- Data & Retrieval Setup ---
CSV_PATH = os.path.join(os.path.dirname(__file__), '13k-recipes.csv')
if not os.path.exists(CSV_PATH):
    df = pd.DataFrame({
        'Title': ['Chicken Stir-fry', 'Lentil Soup', 'Pasta Salad'],
        'Ingredients': ['Chicken, Bell Pepper, Onion, Tomato, Spices', 'Lentil, Onion, Carrot, Tomato Paste, Mint', 'Pasta, Mayonnaise, Peas, Carrot'],
        'Instructions': ['Stir-fry chicken...', 'Boil lentils...', 'Boil pasta...'],
        'Cleaned_Ingredients': ['chicken bell pepper onion tomato spices', 'lentil onion carrot tomato paste mint', 'pasta mayonnaise peas carrot']
    })
else:
    df = pd.read_csv(CSV_PATH)
    expected_cols = ['Title', 'Ingredients', 'Instructions', 'Cleaned_Ingredients']
    for c in expected_cols:
        if c not in df.columns:
            raise ValueError(f"CSV dosyasında '{c}' sütunu bulunamadı.")
    df = df[expected_cols].dropna().reset_index(drop=True)

vectorizer = TfidfVectorizer()
if not df.empty:
    X = vectorizer.fit_transform(df['Cleaned_Ingredients'].astype(str))
else:
    X = None
    print("Uyarı: DataFrame boş, TF-IDF vektörleyicisi oluşturulamadı.")

def retrieve_recipes(user_input_en, top_n=3):
    """Kullanıcı girdisine benzer top_n tarifi (pandas DataFrame) döndürür."""
    if X is None or df.empty:
        return pd.DataFrame() # Veri yoksa boş DataFrame dön
    
    input_vec = vectorizer.transform([user_input_en])
    similarity = cosine_similarity(input_vec, X).flatten()
    
    min_similarity = 0.01  # Gerekirse benzerlik eşiğini ayarlayabilirsiniz
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
        print(f"[GEMINI HATA] {e}")
        return f"Üzgünüm, Gemini API ile iletişim kurarken bir sorun oluştu: {str(e)}"

# --- HTML Template with updated chat interface ---
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Yemek Tarifi Asistanı</title>
    <link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Open+Sans:wght@400;600&display=swap" rel="stylesheet">
    <style>
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

        body {
            margin: 0;
            padding: 0;
            font-family: 'Open Sans', sans-serif;
            background-color: var(--background-color);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .chat-container {
            width: 100%;
            max-width: 750px;
            background: var(--chat-bg);
            border-radius: 18px;
            box-shadow: 0 12px 35px rgba(0,0,0,0.15);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            height: 90vh;
            margin: 20px;
        }
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
        }
        .chat-messages {
            flex-grow: 1;
            padding: 20px;
            overflow-y: auto;
            background-color: var(--chat-bg);
            scroll-behavior: smooth;
        }
        .message {
            margin-bottom: 18px;
            display: flex;
            align-items: flex-start;
        }
        .user-message {
            justify-content: flex-end;
        }
        .bot-message {
            justify-content: flex-start;
        }
        .message-content {
            padding: 14px 20px;
            border-radius: 25px;
            max-width: 75%;
            line-height: 1.6;
            font-size: 1em;
            word-wrap: break-word;
            white-space: pre-wrap;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .user-message .message-content {
            background: var(--user-bubble-bg);
            color: white;
            border-bottom-right-radius: 8px; /* Tail effect */
        }
        .bot-message .message-content {
            background: var(--bot-bubble-bg);
            color: var(--text-color);
            border: 1px solid var(--border-color);
            border-bottom-left-radius: 8px; /* Tail effect */
        }
        .chat-input {
            padding: 18px 25px;
            border-top: 1px solid var(--border-color);
            background-color: var(--input-bg);
        }
        .input-container {
            display: flex;
            gap: 12px;
            align-items: center;
        }
        #user-input {
            flex-grow: 1;
            padding: 14px 20px;
            border: 1px solid var(--border-color);
            border-radius: 28px;
            outline: none;
            font-size: 1em;
            transition: border-color 0.3s, box-shadow 0.3s;
            background-color: var(--input-bg);
            color: var(--text-color);
        }
        #user-input:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 4px rgba(160, 82, 45, 0.2);
        }
        button {
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
        }
        button:hover {
            background: linear-gradient(90deg, var(--primary-color), #d2691e); /* Chocolate-like hover */
            transform: translateY(-2px);
            box-shadow: 0 7px 20px rgba(0,0,0,0.2);
        }
        button:active {
            transform: translateY(0);
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }
        .loading {
            display: none;
            text-align: center;
            padding: 12px;
            color: var(--primary-color);
            font-style: italic;
            background-color: var(--background-color);
            border-top: 1px solid var(--border-color);
        }
        /* Custom styles for recipe output */
        .message-content h3 {
            font-family: 'Merriweather', serif;
            color: var(--primary-color);
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            font-size: 1.3em;
            font-weight: 700;
        }
        .message-content p {
            margin-bottom: 0.8em;
        }
        .message-content b {
            font-family: 'Merriweather', serif;
            color: var(--primary-color);
            display: block; /* Ensures bold labels are on their own line */
            margin-top: 1em;
            margin-bottom: 0.5em;
            font-size: 1.1em;
        }
        .message-content ul, .message-content ol {
            margin-left: 20px;
            margin-bottom: 1em;
            padding: 0;
            list-style-position: inside; /* Keeps bullets/numbers aligned */
        }
        .message-content ul li {
            list-style-type: '🍪 '; /* Custom list icon */
            margin-bottom: 0.5em;
        }
        .message-content ol li {
            margin-bottom: 0.5em;
        }

        /* Responsive adjustments */
        @media (max-width: 600px) {
            .chat-container {
                height: 95vh;
                margin: 10px;
                border-radius: 12px;
            }
            .chat-header {
                font-size: 1.5em;
                padding: 18px;
            }
            .message-content {
                max-width: 85%;
                font-size: 0.9em;
                padding: 12px 16px;
                border-radius: 20px;
            }
            .chat-input {
                padding: 12px 18px;
            }
            #user-input {
                padding: 12px 16px;
                font-size: 0.9em;
                border-radius: 22px;
            }
            button {
                padding: 12px 22px;
                font-size: 0.95em;
                border-radius: 22px;
            }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            Lezzetli Tarif Asistanı
        </div>
        <div class="chat-messages" id="chat-messages">
            <div class="message bot-message">
                <div class="message-content">
                    Merhaba! Ben Lezzetli Tarif Asistanınızım. Mutfakta harikalar yaratmak için hangi malzemelerle yola çıkmak istersiniz? 😊
                </div>
            </div>
        </div>
        <div class="loading" id="loading">Tarifleri düşünüyorum, birazdan hazırlarım...</div>
        <div class="chat-input">
            <div class="input-container">
                <input type="text" id="user-input" placeholder="Malzemeleri yazın (örn: tavuk, mantar, krema)...">
                <button onclick="sendMessage()">Tarif Bul</button>
            </div>
        </div>
    </div>

    <script>
        function addMessage(message, isUser) {
            const messagesDiv = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            // Use innerHTML because Gemini will return HTML-formatted text
            messageDiv.innerHTML = `<div class="message-content">${message}</div>`;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            
            if (message) {
                addMessage(message, true);
                input.value = '';
                showLoading(true);

                try {
                    const response = await fetch('/api/get_recipe', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ user_input: message }) // top_n will be controlled server-side
                    });

                    const data = await response.json();
                    showLoading(false);
                    
                    if (data.error) {
                        addMessage('Üzgünüm, bir hata oluştu: ' + data.error, false);
                    } else {
                        addMessage(data.generated, false);
                    }
                } catch (error) {
                    showLoading(false);
                    console.error('Fetch error:', error);
                    addMessage('Üzgünüm, sunucuya ulaşırken bir hata oluştu. Lütfen internet bağlantınızı kontrol edip tekrar deneyin.', false);
                }
            }
        }

        document.getElementById('user-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/get_recipe', methods=['POST'])
def api_get_recipe():
    try:
        data = request.get_json(silent=True) or {}
        user_input_tr = (data.get('user_input') or '').strip() 
        # top_n is now controlled by the backend logic, default 1, max 3
        requested_top_n = data.get('top_n', 1) # Default to 1 recipe if not specified
        top_n_for_retrieval = max(1, min(int(requested_top_n), 3)) # Ensure it's between 1 and 3

        if not user_input_tr:
            return jsonify({'error': 'Lütfen malzemeleri belirtin.'}), 400

        # --- İlk olarak, kullanıcının girdisinin yemek tarifiyle ilgili olup olmadığını kontrol edelim ---
        # Gemini'ye genel bir kontrol prompt'u gönderiyoruz
        # Bu prompt'u daha karmaşık hale getirip farklı niyetleri (intent) anlayabilirsiniz.
        intent_prompt = f"Kullanıcının isteği: \"{user_input_tr}\". Bu istek yemek tarifi veya malzeme sorgusu ile ilgili mi? Sadece 'EVET' veya 'HAYIR' olarak yanıt ver."
        intent_response = call_gemini(intent_prompt)
        
        if "HAYIR" in intent_response.upper():
            return jsonify({'generated': "Üzgünüm, ben sadece yemek tarifleri konusunda yardımcı olabiliyorum. Lütfen yemekle ilgili bir soru sorun. 🧑‍🍳"}), 200

        user_input_en = user_input_tr
        if TRANSLATOR_AVAILABLE:
            try:
                translated_text = translator.translate(user_input_tr, src='tr', dest='en').text
                user_input_en = translated_text
                print(f"[TRANSLATION] TR: '{user_input_tr}' -> EN: '{user_input_en}'")
            except Exception as e:
                print(f"[TRANSLATION ERROR] Could not translate: {e}. Using original input for retrieval.")
        else:
            print("[TRANSLATION] Translator not available. Proceeding with original (Turkish) input for retrieval.")
        
        # Retrieve recipes using the (translated) English input
        recipes = retrieve_recipes(user_input_en, top_n=top_n_for_retrieval)
        
        # Convert recipes to list for JSON response
        retrieved_list = recipes.to_dict(orient='records') if not recipes.empty else []
        
        context = ""
        if recipes.empty:
            # Eğer eşleşen tarif bulunamazsa Gemini'ye bu durumu bildiren bir bağlam gönderiyoruz.
            context = "Veritabanında kullanıcının belirttiği malzemelerle doğrudan eşleşen bir tarif bulunamadı."
        else:
            rows = []
            for _, row in recipes.iterrows():
                # HTML etiketlerini doğrudan Gemini'nin çıktısında olmasını istediğimiz formatta ayarlıyoruz.
                rows.append(
                    f"<h3>{row['Title']}</h3>"
                    f"<p>{row['Ingredients']}</p>" # Description could go here if available, or a short intro
                    f"<b>Malzemeler:</b>\n<ul><li>" + "</li><li>".join(row['Ingredients'].split(', ')) + "</li></ul>\n"
                    f"<b>Yapılışı:</b>\n<ol><li>" + "</li><li>".join(re.split(r'\d+\.\s*', row['Instructions'])[1:]) + "</li></ol>"
                )
            context = "\n<hr>\n".join(rows) # Tarifleri ayırmak için HR kullanabiliriz

        # Generate prompt for Gemini
        # Prompt'u, Gemini'den istediğimiz çıktının formatını HTML etiketleri kullanarak belirtecek şekilde güncelliyoruz.
        # Ayrıca, eğer tarif bulunamadıysa ne yapması gerektiğini de belirtiyoruz.
        if recipes.empty:
            prompt = f"Kullanıcı şu malzemeleri belirtti: \"{user_input_tr}\"\n\n" \
                     f"Mevcut tarifler:\n{context}\n\n" \
                     f"Veritabanında bu malzemelerle eşleşen bir tarif bulunamadığı için, lütfen kullanıcıya bu durumu bildir ve bu malzemelerle hazırlanabilecek genel bir tarif fikri veya alternatif bir öneri sun. Yanıtını Türkçe, samimi ve motive edici bir dille ver. Lütfen HTML etiketleri (h3, b, ul, li, ol) kullanarak formatla."
        else:
            prompt = f"Kullanıcı şu malzemeleri belirtti: \"{user_input_tr}\"\n\n" \
                     f"Aşağıdaki mevcut tarifleri referans alarak kullanıcının malzemelerine en uygun tarifleri öner. Eğer tarifi aynen kullanırsan, lütfen verilen yapıyı koru. Eğer yeni bir fikir üretirsen, benzer formatta bir tarif oluştur.\n\n" \
                     f"Mevcut Tarifler:\n{context}\n\n" \
                     f"Lütfen yanıtını Türkçe, samimi ve motive edici bir dille ver. Tarifleri HTML etiketleri (h3 başlık için, b etiketleri 'Malzemeler:' ve 'Yapılışı:' için, ul/li malzeme listesi için, ol/li yapılış adımları için) kullanarak formatla. Gerekirse kısa ve sevimli bir giriş cümlesi kullanabilirsin. Örneğin: 'Harika bir seçim! Bu malzemelerle neler yapabiliriz bakalım...'" \
                     f"Sadece {top_n_for_retrieval} adet tarif öner."

        generated_response = call_gemini(prompt)
        
        return jsonify({'generated': generated_response, 'retrieved_recipes': retrieved_list})
    
    except Exception as e:
        print(f"[API] Error in api_get_recipe: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    configure_genai()
    app.run(debug=True)