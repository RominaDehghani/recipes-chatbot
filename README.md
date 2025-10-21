<img width="1915" height="962" alt="Ekran görüntüsü 2025-10-21 225540" src="https://github.com/user-attachments/assets/18e57f0f-793c-499c-b843-0918c7d81c58" />


https://github.com/user-attachments/assets/ee3ee537-ce89-4d2e-9b2d-52308a3aec81


# 🍲 Delicious Recipe Assistant

This project implements an AI-powered recipe recommendation system utilizing Streamlit for the frontend, a pre-processed recipe dataset, and the Google Gemini API for intent recognition and generative recipe suggestions.

## 🎯 Project Objective

The primary objective is to provide users with efficient and relevant recipe recommendations based on provided ingredients. The system aims to enhance kitchen productivity by:

*   Facilitating quick discovery of recipes matching available ingredients.
*   Leveraging generative AI to propose novel or adapted recipe ideas.
*   Streamlining the meal planning and cooking process.

## 📊 Dataset

The core of the recommendation system relies on a `13k-recipes.csv` dataset. This dataset is structured with the following key attributes:

*   **Title**: Recipe name.
*   **Ingredients**: Comma-separated list of primary ingredients.
*   **Instructions**: Step-by-step preparation guidelines.
*   **Cleaned_Ingredients**: A pre-processed, tokenized, and lowercased version of ingredients used for vectorization.

**Note**: In the absence of `13k-recipes.csv`, the application gracefully degrades to a smaller, in-memory sample dataset for continued functionality.

## ⚙️ Methodology and Technologies

The system integrates several components and techniques:

1.  **Frontend (Streamlit)**: Provides an interactive, chatbot-like web interface. Custom CSS is applied for aesthetic enhancements and chat bubble alignment.
2.  **Natural Language Processing (NLP)**:
    *   **TF-IDF Vectorization (`TfidfVectorizer`)**: Transforms `Cleaned_Ingredients` from the dataset and user input into numerical TF-IDF feature vectors. This enables semantic comparison of ingredient lists.
    *   **Cosine Similarity**: Computes the cosine similarity between the user's input vector and the TF-IDF matrix of the recipe dataset (`X`), identifying the most semantically similar recipes.
3.  **Google Gemini API (`gemini-2.5-flash`)**:
    *   **Intent Recognition**: A prompt-based approach is used to classify user input as either a "recipe/ingredient query" or not. This ensures the assistant responds appropriately to non-recipe-related questions. The check involves looking for both "NO" and absence of "YES" in the Gemini response to improve robustness.
    *   **Generative Recipe Formulation**: When the retrieval component yields no direct matches, or when detailed guidance is required, Gemini generates a recipe by synthesizing information based on the user's input and optionally, the nearest available recipes. The output is formatted with HTML tags (`<h3>`, `<b>`, `<ul>`, `<ol>`) for structured presentation.
4.  **Data Handling (Pandas)**: Utilized for efficient loading, cleaning, and manipulation of the recipe dataset.
5.  **Environment Management (`dotenv`)**: Securely loads API keys and other sensitive configurations from a `.env` file.

## ✨ Key Features and Outcomes

*   **Robust Intent-Based Interaction**: Proactively filters non-recipe-related queries, preventing irrelevant responses and maintaining conversational focus.
*   **Hybrid Recommendation Engine**: Combines a TF-IDF/Cosine Similarity-based retrieval system with a large language model (LLM) for both exact matching and generative capabilities.
*   **Dynamic Content Generation**: Leverages Gemini's generative power to create contextually relevant recipe suggestions, even for novel ingredient combinations.
*   **Structured Output**: Recipes are presented in a well-formatted HTML structure, improving readability and user comprehension.
*   **Performance Optimization**: Utilizes Streamlit's `@st.cache_data` decorator for efficient data loading and vectorizer initialization, coupled with a fast Gemini model for low-latency API calls.
*   **Fault Tolerance**: Includes graceful fallback mechanisms (e.g., sample dataset if CSV is missing, mock API response if Gemini is unavailable).

This project exemplifies the integration of traditional NLP techniques with state-of-the-art generative AI to build a practical and intelligent assistant.

## ⚙️ Installation & Setup in Local

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/recipe-chatbot.git
cd recipe-chatbot
```

### Step 2: Install Dependencies
Install only the necessary packages from requirements.txt:
```bash
pip install -r requirements.txt
```

### Step 3: Set Up Environment Variables
Create a file named .env in the project root directory and add your API key:
```bash
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### Step 4: Run the Application
If you’re using Streamlit:
```bash
streamlit run app.py
```
Or, if you’re using a Python CLI app:
```bash
python app.py
```

## 🌐 Live Demo
👉 (https://recipes-chatbot-nlkrndrjyqqrcotgnef7wz.streamlit.app/)
