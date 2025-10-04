# 🧠 CV–JD Matching System using NER and BERT

This project automates the process of **matching resumes (CVs)** with **job descriptions (JDs)** using advanced **Natural Language Processing (NLP)** techniques.

It extracts **skills** using a custom **Named Entity Recognition (NER)** model (built with spaCy), and evaluates similarity between candidates and job descriptions using **DistilBERT embeddings** and **cosine similarity**.

---

## 🚀 Features

- **PDF/Text Extraction:** Reads CVs and job descriptions in `.pdf` or `.txt` format.
- **Skill Extraction:** Uses a custom-trained spaCy NER model to detect skills.
- **Semantic Matching:** Uses DistilBERT embeddings for context-aware similarity scoring.
- **Similarity Analytics:** Visualizes CV–JD similarity scores using Matplotlib.
- **Interactive Web Interface:** Built with Streamlit for easy use.
- **Batch Processing:** Handles multiple CVs at once and ranks them by similarity.

---

## 🧰 Technologies Used

| Category          | Tools / Libraries                                          |
| ----------------- | ---------------------------------------------------------- |
| **Language**      | Python 3.10+                                               |
| **Frameworks**    | Streamlit, spaCy, Transformers (Hugging Face)              |
| **NLP Models**    | Custom spaCy NER, DistilBERT                               |
| **Libraries**     | pandas, numpy, pdfplumber, scikit-learn, torch, matplotlib |
| **Visualization** | Matplotlib                                                 |
| **Deployment**    | Streamlit app                                              |

---

## 🧾 How It Works

### 1️⃣ Extract Text

Uses `pdfplumber` to extract clean text from PDFs and stores it for further processing.

### 2️⃣ Extract Skills

Loads a trained **spaCy NER model (`model-best`)** and detects skill entities labeled as `SKILL`.

### 3️⃣ Embed and Compare

Uses **DistilBERT (`distilbert-base-uncased`)** to generate embeddings and computes **cosine similarity** between JD and each CV skill set.

### 4️⃣ Rank Candidates

Sorts candidates based on similarity scores (higher = better match).

---

## Run the app

streamlit run app.py

Then open the displayed local URL (e.g. http://localhost:8501) in your browser.

## 📈 Example Output

After uploading CVs and a job description:

## Results:

CV_John.pdf → Similarity: 0.87
CV_Emily.pdf → Similarity: 0.72
CV_Rahul.pdf → Similarity: 0.65

A bar chart displays the similarity scores for visual comparison.

## 🧩 Future Enhancements

1. Improve NER with more labeled data for better skill detection

2. Integrate additional metadata (education, experience)

3. Add web-based deployment via Streamlit Cloud or Hugging Face Spaces

4. Export top-matching candidates automatically
