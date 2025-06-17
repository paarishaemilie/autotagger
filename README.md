# AutoTagger

A Streamlit web app that automatically extracts blog content from a given URL and suggests SEO-friendly tags using NLP-based techniques such as frequency analysis, RAKE (Rapid Automatic Keyword Extraction), and KeyBERT. The app performs thorough content cleaning, lemmatization, and displays visual insights like word clouds and top keyword tables — ideal for bloggers, digital writers, and content marketers.

## 🧠 What It Does

AutoTagger solves a common content marketing pain point: choosing effective tags for your blog posts. Instead of guessing or copying others, this app intelligently reads your post and suggests data-driven tags based on actual word relevance, semantic meaning, and keyword frequency.

## 🔍 Features

- 🌐 Extracts text directly from a live blog URL (article, post body, or page)
- 🧹 Tokenizes, cleans, and lemmatizes raw content for analysis
- 📊 Shows total and cleaned word counts for transparency
- ☁️ Displays a word cloud based on cleaned tokens
- 📈 Suggests blog tags using 3 distinct methods:
  - **Word Frequency**: Top repeated, meaningful terms
  - **RAKE**: Extracts ranked keyword phrases (1–2 words)
  - **KeyBERT**: Uses BERT embeddings to extract semantically relevant keywords
- 🧾 Tabulates and highlights the top 10 valid tags from each method

## ⚙️ Tech Stack

- **Streamlit**: UI and interactive input handling
- **BeautifulSoup**: HTML parsing and content extraction
- **NLTK**: Text preprocessing, lemmatization, stopword filtering
- **RAKE-NLTK**: Extracts keyword phrases from raw text
- **KeyBERT + Sentence-Transformers**: Semantic keyword generation
- **WordCloud + Matplotlib**: Data visualization
- **Python Standard Libraries**: `re`, `collections`, `requests`

## 🖼️ Screenshot

<details>
  <summary>Click to view screenshot</summary>

  <br>

  ![AutoTagger Screenshot](https://github.com/paarishaemilie/autotagger/blob/main/app.png)

</details>

## 🚀 Live App

You can try the app here:  
👉 https://autotaggerforblog.streamlit.app

## 📦 Installation

### Clone the Repository

```bash
git clone https://github.com/paarishaemilie/autotagger.git
cd autotagger
```

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure you also download NLTK data files (these are already triggered in app.py):
```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## 🚀 Run the App
```bash
streamlit run app.py
```
Then open your browser at:
👉 http://localhost:8501

## 📁 Project Structure
```bash
AutoTagger/
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependency list
├── README.md             # Project documentation
```

## 📚 Dependencies
- streamlit
- beautifulsoup4
- requests
- nltk
- rake-nltk
- keybert
- sentence-transformers
- scikit-learn
- wordcloud
- matplotlib
- numpy
- pandas

## 👤 Author
Built with 🧡 by @paarishaemilie
