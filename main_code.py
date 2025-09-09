import streamlit as st
import pdfplumber
import os
import spacy
import pandas as pd
import torch
import time
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load pre-trained spaCy and DistilBERT models
nlp = spacy.load("model-best")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def save_file(uploaded_file):
    """ Saves the uploaded file to disk. """
    if not os.path.exists('uploaded_files'):
        os.makedirs('uploaded_files')
    file_path = os.path.join('uploaded_files', uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def read_document(file_path):
    """ Extracts text from a PDF or text file based on the extension. """
    text = ""
    if file_path.endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + ' '
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    return text

def extract_skills(text):
    """ Extracts skills from text using NER spaCy model. """
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ == 'SKILL']
    return skills

def embed_text(texts):
    """ Converts texts to DistilBERT embeddings. """
    embeddings = []
    for text in texts:
        tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output = model(**tokens)
        embeddings.append(output.last_hidden_state.mean(dim=1).numpy()[0])
    return embeddings

def match_cvs(uploaded_files, job_description_file):
    """ Matches CVs against the job description using DistilBERT embeddings and cosine similarity. """
    jd_path = save_file(job_description_file)
    job_description_text = read_document(jd_path)
    job_skills = ' '.join(extract_skills(job_description_text))
    documents = [job_skills]
    filenames = ['Job Description']
    skills_data = {filenames[0]: job_skills}

    for uploaded_file in uploaded_files:
        file_path = save_file(uploaded_file)
        cv_text = read_document(file_path)
        extracted_skills = extract_skills(cv_text)
        skill_text = ' '.join(extracted_skills)
        documents.append(skill_text)
        filenames.append(uploaded_file.name)
        skills_data[uploaded_file.name] = skill_text

    document_embeddings = embed_text(documents)
    cosine_similarities = cosine_similarity([document_embeddings[0]], document_embeddings[1:]).flatten()
    results = list(zip(filenames[1:], cosine_similarities))
    results.sort(key=lambda x: x[1], reverse=True)
    return results, skills_data

def main():
    
    menu = ['Home', 'Instructions', 'Upload Documents', 'Start Matching', 'Results', 'Match Analytics', 'View Extracted Skills']
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Home':
        with st.spinner('Loading...'):
            time.sleep(1)  # simulate some loading process
            col1, col2, col3 = st.columns([1,6,1])
            with col2:
                st.title('CV-JD Matching System with NER')
                st.subheader("Welcome to the CV-JD Matching System")
                st.image("logo.png")
                st.markdown("""
                    ## What is CV-JD Matching System?
                    The **CV-JD Matching System** is designed to automate the process of finding the best candidates for a job based on the job description provided. Using advanced NER and machine learning techniques, the system extracts skills from CVs and job descriptions, then matches them effectively.
                    
                    **Get started by navigating using the sidebar!**
                    """)

    elif choice == 'Instructions':
        st.subheader("Instructions")
        st.markdown("""
            Follow these steps to use the CV-JD Matching System:
            1. **Upload Documents**: Go to the 'Upload Documents' section to upload your CVs and job description.
            2. **Start Matching**: Once documents are uploaded, navigate to 'Start Matching' to begin the process.
            3. **View Results**: After the matching process, go to 'Results' to see the top candidates.
            4. **View Extracted Skills**: Check 'View Extracted Skills' to see the skills extracted from each document.
            """)

    elif choice == 'Upload Documents':
        st.subheader("Upload CVs and Job Description")
        uploaded_files = st.file_uploader("Choose CV files", accept_multiple_files=True, type=['pdf', 'txt'], key='cv_files')
        job_description_file = st.file_uploader("Upload Job Description", type=['pdf', 'txt'], key='jd_file')
        st.session_state['uploaded_files'] = uploaded_files
        st.session_state['job_description_file'] = job_description_file

    elif choice == 'Start Matching':
        st.subheader("Match CVs to Job Description")
        if st.button("Start Matching Process"):
            if st.session_state.get('uploaded_files') and st.session_state.get('job_description_file'):
                with st.spinner('Matching in progress...'):
                    progress_bar = st.progress(0)
                    total_steps = len(st.session_state['uploaded_files']) + 1  # +1 for job description processing
                    step = 0
                    jd_path = save_file(st.session_state['job_description_file'])
                    job_description_text = read_document(jd_path)
                    job_skills = ' '.join(extract_skills(job_description_text))
                    documents = [job_skills]
                    filenames = ['Job Description']
                    skills_data = {filenames[0]: job_skills}
                    progress_bar.progress((step + 1) / total_steps)

                    for uploaded_file in st.session_state['uploaded_files']:
                        step += 1
                        file_path = save_file(uploaded_file)
                        cv_text = read_document(file_path)
                        extracted_skills = extract_skills(cv_text)
                        skill_text = ' '.join(extracted_skills)
                        documents.append(skill_text)
                        filenames.append(uploaded_file.name)
                        skills_data[uploaded_file.name] = skill_text
                        progress_bar.progress((step + 1) / total_steps)

                    document_embeddings = embed_text(documents)
                    cosine_similarities = cosine_similarity([document_embeddings[0]], document_embeddings[1:]).flatten()
                    results = list(zip(filenames[1:], cosine_similarities))
                    results.sort(key=lambda x: x[1], reverse=True)
                    st.session_state['results'] = results
                    st.session_state['skills_data'] = skills_data
                    st.success("Matching complete!")
                    progress_bar.empty()
            else:
                st.error("Please upload all required files.")

    elif choice == 'Results':
        st.subheader("Results")
        st.subheader("View Top Candidates")
        if 'results' in st.session_state:
            results_df = pd.DataFrame(st.session_state['results'], columns=['CV', 'Similarity Score'])
            st.write(results_df)
        else: 
            st.warning("No results to show. Please run the matching process first.")
    
    elif choice == 'Match Analytics':
        st.subheader("Match Analytics")
        if 'results' in st.session_state and st.session_state['results']:
            results_df = pd.DataFrame(st.session_state['results'], columns=['CV', 'Similarity Score'])
            
            if not results_df.empty:
                fig, ax = plt.subplots()
                ax.barh(results_df['CV'], results_df['Similarity Score'], color='skyblue')
                ax.set_xlabel('Similarity Score')
                ax.set_title('CV to Job Description Similarity')
                st.pyplot(fig)
            else:
                st.warning("No similarity scores available. Please run some matches first.")
        else:
            st.warning("No matching results available. Please complete the matching process first to generate data for analytics.")


    elif choice == 'View Extracted Skills':
        st.subheader("Extracted Skills from Documents")
        if 'skills_data' in st.session_state:
            for filename, skills in st.session_state['skills_data'].items():
                st.subheader(f"Skills from {filename}")
                st.text(skills)
        else:
            st.warning("No skills data available. Please complete the matching process first.")

if __name__ == "__main__":
    main()
