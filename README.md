---

# 🎬 Movie Recommendation System (Content-Based)

An end-to-end **content-based movie recommendation system** built using **Word2Vec embeddings** and deployed as an interactive **Streamlit web application**.
The system recommends movies based on **semantic similarity** of movie metadata and enhances user experience with **movie posters fetched via the TMDB API**.

---

## 📌 Project Overview

Recommendation systems are widely used in real-world applications such as Netflix and Amazon to personalize content.
This project implements a **content-based recommender**, where movies are recommended based on their **content similarity**, not user ratings.

Each movie is represented as a **dense vector embedding**, and recommendations are generated using **cosine similarity**.

---

## 🧠 Approach & Methodology

### 1️⃣ Dataset & Feature Engineering

* Dataset: **TMDB 5000 Movies Dataset**
* Metadata used:

  * Movie overview (plot)
  * Genres
  * Keywords
  * Cast
  * Director
* All metadata is combined into a single **tags** column
* Text preprocessing:

  * Lowercasing
  * Tokenization
  * Stopword removal
  * Noise cleaning

---

### 2️⃣ Word2Vec Embedding

* A **custom Word2Vec model** is trained using Gensim
* Each word is converted into a dense vector
* A movie vector is created by **averaging its word embeddings**
* This allows the model to capture **semantic meaning**, not just keyword overlap

📌 **Why Word2Vec?**

* Captures contextual similarity
* Performs better than TF-IDF for semantic recommendations
* Handles different vocabulary with similar meaning

---

### 3️⃣ Similarity Computation

* Cosine similarity is used to measure similarity between movie vectors
* Top-N most similar movies are recommended for a given input movie

---

### 4️⃣ Training vs Inference Separation

* Model training and vector generation are done **offline**
* Saved artifacts are reused during inference for **fast performance**

Saved files:

* `movies.pkl` → Movie metadata
* `word2vec.model` → Trained Word2Vec model
* `movie_vectors.npy` → Precomputed movie embeddings

---

## 🖥️ Web Application (Streamlit)

The recommender system is deployed as a **Streamlit web app**.

### ✨ Features

* 🔍 Search-based movie selection
* 🎬 Button-controlled recommendations
* 🎯 Adjustable Top-N recommendations
* 🖼️ Movie posters fetched via TMDB API
* ⏳ Loading spinners for better UX
* ℹ️ About section explaining system design
* 🔐 Secure API key handling using environment variables

---

## 🧰 Tech Stack

| Category      | Tools                           |
| ------------- | ------------------------------- |
| Language      | Python                          |
| ML / NLP      | Gensim (Word2Vec), Scikit-learn |
| Data Handling | Pandas, NumPy                   |
| Web Framework | Streamlit                       |
| External API  | TMDB API                        |
| Environment   | python-dotenv                   |

---

## 📂 Project Structure

```
MOVIE-RECOMMENDATION-SYSTEM/
│
├── app.py
├── movies.pkl
├── movie_vectors.npy
├── word2vec.model
├── requirements.txt
├── README.md
├── .env        (ignored)
├── .gitignore
└── venv/       (local environment)
```

---

## 🔐 API Key Security

* TMDB API key is stored in a `.env` file
* Environment variables are loaded using `python-dotenv`
* `.env` is excluded from Git tracking to prevent key leaks

Example `.env` file:

```env
TMDB_API_KEY=your_api_key_here
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Vedshinde06/Movie-Recommender-System.git
cd movie-recommendation-system
```

### 2️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add TMDB API Key

Create a `.env` file in the project root:

```env
TMDB_API_KEY=your_api_key_here
```

### 5️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Example Output

**Input:** Batman Begins
**Recommended Movies:**

* The Dark Knight
* Batman Returns
* Man of Steel
* Watchmen
* V for Vendetta

(Posters displayed dynamically using TMDB API)

---

## 🧪 Limitations & Future Improvements

### Current Limitations

* Cold-start problem for unseen movies
* No user personalization (pure content-based)

### Future Enhancements

* Hybrid recommendation (content + collaborative)
* Use pretrained embeddings (GloVe / BERT)
* Genre or mood-based filters
* Improved poster matching using TMDB movie IDs

---

## 🧠 Key Learnings

* Practical implementation of **Word2Vec embeddings**
* Designing ML systems with **training–inference separation**
* Handling external APIs reliably
* Building **production-ready ML applications**
* Combining ML with user-focused UI design

---

## 👤 Author

**Vedant Shinde**
Artificial Intelligence & Data Science
Pune, India

---

## ⭐ Final Note

This project demonstrates a **full ML lifecycle** — from data preprocessing and model training to deployment and user interaction.

---

