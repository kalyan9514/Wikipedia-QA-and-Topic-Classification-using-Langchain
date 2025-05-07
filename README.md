# Wikipedia-QA-and-Topic-Classification-using-Langchain

**Project Overview:**

This project aims to build a system that answers user questions by extracting relevant responses from Wikipedia content and classifying them into appropriate topics. By combining question answering and topic classification, the system enhances information retrieval accuracy and user experience. The solution uses semantic similarity techniques and multiple embedding models (Word2Vec, GloVe, FastText) along with deep learning models such as LSTM, Siamese Networks, and BERT.

**Files Included:**

Code.ipynb – The main implementation notebook with all models, preprocessing, and analysis.

wiki_qa_df.csv – Preprocessed dataset containing questions, answers, and topic labels.

**Dataset Summary:**

- Dataset: https://huggingface.co/datasets/microsoft/wiki_qa

- Source: Microsoft Research

- Questions: 3,047

- Candidate Answers: 29,000+

- Structure: Questions paired with potential answers from Wikipedia articles, labeled as relevant or not

**Key Columns:**

| Column Name      | Description                    |
| ---------------- | ------------------------------ |
| `question`       | The user question              |
| `document_title` | Wikipedia article title/topic  |
| `answer`         | Candidate answer sentence      |
| `label`          | 1 if relevant, 0 if irrelevant |

**Workflow Overview:**

1. Data Preprocessing
- Removal of stopwords
- Lemmatization
- Text cleaning (punctuation, casing)
- Tokenization

2. Feature Engineering
- Embeddings using Word2Vec, GloVe, and FastText
- Cosine similarity computation between Q&A pairs
- TF-IDF vectorization

3. Model Development
- Baseline & Classical Models:
- TF-IDF Similarity
- Cosine Similarity with embeddings

4. Deep Learning Models:
- LSTM
- Siamese Network
- BERT (fine-tuned on WikiQA)

**Model Performance Comparison:**

| Model               | Accuracy (%) |
| ------------------- | ------------ |
| **Siamese Network** | **96.85%**   |
| BERT                | 94.24%       |
| TF-IDF              | 75.12%       |
| GloVe Baseline      | 70.54%       |
| LSTM                | 29.29%       |

**How to Run the Project:**

1. Clone the Repository

```
git clone https://github.com/your-username/WikiQA-QA-Classification.git
cd WikiQA-QA-Classification
```

2. Install Required Libraries

```
pip install pandas numpy nltk gensim sklearn matplotlib seaborn tensorflow transformers
```

3. Launch the Notebook

```
jupyter notebook Final_Code.ipynb
```

**Key Features:**

- Handles semantic similarity using multiple embeddings

- Supports exact match and similarity-based QA retrieval

- Visual EDA (length analysis, word clouds, correlation matrices)

- Text classification using BERT and Siamese Network

- Evaluation on test and unseen questions

**Future Enhancements:**

- Train on larger QA datasets (e.g., SQuAD, CoQA)

- Fine-tune semantic matching using contrastive learning

- Build an interactive chatbot UI

- Extend to multilingual QA via machine translation


