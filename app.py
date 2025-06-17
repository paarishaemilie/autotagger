import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rake_nltk import Rake
from keybert import KeyBERT
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

# --- Download NLTK assets if not already present ---
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

download_nltk_data()

# --- Initialize ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

st.title("🏷️ AutoTagger: Suggest Blog Tags by URL")
url = st.text_input("Paste your blogpost URL:")

# --- Clean tag helper ---
def clean_tag(tag):
    tag = tag.lower()
    tag = re.sub(r'[^\w\s-]', '', tag)  # remove special chars
    tag = re.sub(r'[\u200b\u00a0–—]+', '', tag)  # remove invisible chars/dashes
    tag = tag.strip("-—– ").strip()
    return tag

# --- Lemmatize, deduplicate and ensure 10 entries ---
def lemmatize_and_dedup_extended(tag_list, limit=10):
    seen = set()
    final = []
    for tag in tag_list:
        cleaned = clean_tag(tag)
        if len(cleaned) <= 1:
            continue
        lemma = " ".join([lemmatizer.lemmatize(w) for w in cleaned.split()])
        if lemma not in seen:
            seen.add(lemma)
            final.append((cleaned, lemma))  # (display, logic)
        if len(final) == limit:
            break
    return final

# --- Remove duplicates helper ---
def remove_duplicates_preserve_order(tag_list):
    seen = set()
    return [tag for tag in tag_list if not (tag in seen or seen.add(tag))]

# --- Clean and extract text from blog page ---
def extract_clean_text(url):
    try:
        page = requests.get(url, timeout=10)
        if page.status_code >= 400:
            return None
        soup = BeautifulSoup(page.content, "html.parser")
        article = soup.find('article') or soup.find('div', class_="post-content") or soup.body
        text = article.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        error_keywords = ["error", "server", "not found", "acceptable", "resource", "requested", "appropriate", "representation"]
        if any(kw in text.lower() for kw in error_keywords) and len(text.split()) < 100:
            return None
        return text
    except Exception:
        return None

# --- Main processing ---
if url:
    raw_text = extract_clean_text(url)
    if raw_text:
        st.success("✅ Blog content successfully extracted.")

        total_word_count = len(raw_text.split())
        tokens = nltk.word_tokenize(raw_text.lower())
        cleaned_tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]

        st.markdown(f"**📏 Total Word Count (raw):** {total_word_count}")
        st.markdown(f"**🧽 Cleaned Word Count (no stopwords/punctuation):** {len(cleaned_tokens)}")

        wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(cleaned_tokens))
        st.image(wc.to_array(), caption="Word Cloud")

        # ---------- Word Frequency ----------
        with st.expander("🔢 Suggested Tags (Word Frequency Method)"):
            freq_raw = Counter(cleaned_tokens).most_common(100)
            cleaned_freq = [(clean_tag(w), c) for w, c in freq_raw if len(clean_tag(w)) > 1]
            freq_tags = lemmatize_and_dedup_extended([w for w, _ in cleaned_freq], limit=10)
            table_freq = [(tag, dict(cleaned_freq).get(tag, 0)) for tag, _ in freq_tags]

            st.write("Top Words:")
            st.table(table_freq)
            st.markdown("**Suggested Tags:**")
            st.code(", ".join([t[0] for t in freq_tags]), language='markdown')

        # ---------- RAKE ----------
        with st.expander("🧠 Suggested Tags (RAKE Keyword Extraction)"):
            rake = Rake()
            rake.extract_keywords_from_text(raw_text)
            raw_rake = rake.get_ranked_phrases()
            rake_candidates = [
                phrase for phrase in raw_rake
                if 1 <= len(phrase.split()) <= 2 and len(clean_tag(phrase)) > 1
            ]
            rake_candidates = remove_duplicates_preserve_order(rake_candidates)
            rake_tags = lemmatize_and_dedup_extended(rake_candidates, limit=10)
            table_rake = [t[0] for t in rake_tags]

            st.write("Top RAKE Phrases:")
            st.table(table_rake)
            st.markdown("**Suggested Tags:**")
            st.code(", ".join(table_rake), language='markdown')

        # ---------- KeyBERT ----------
        with st.expander("🤖 Suggested Tags (KeyBERT Semantic Keywords)"):
            kw_model = KeyBERT()
            keybert_raw = kw_model.extract_keywords(raw_text, top_n=100, stop_words='english')
            keybert_candidates = [kw[0] for kw in keybert_raw if len(clean_tag(kw[0])) > 1]
            keybert_candidates = remove_duplicates_preserve_order(keybert_candidates)
            keybert_tags = lemmatize_and_dedup_extended(keybert_candidates, limit=10)
            table_keybert = [t[0] for t in keybert_tags]

            st.write("Top KeyBERT Keywords:")
            st.table(table_keybert)
            st.markdown("**Suggested Tags:**")
            st.code(", ".join(table_keybert), language='markdown')

    else:
        st.error("❌ Could not extract meaningful content from the URL. It may be an error page or unsupported format.")
