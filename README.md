# Intelligent Customer Support Ticket System

## NLP-Based Classification, Prioritization & Response Automation

An end-to-end Machine Learning and Natural Language Processing project that analyzes customer support conversations and provides intelligent assistance for ticket classification, prioritization, automated response generation, and response-time prediction.

## Project Objectives

- Automatically identify the intent of a customer query.
- Determine the urgency and priority of a ticket.
- Generate an appropriate customer-support response.
- Predict the expected response time in minutes.
- Reduce manual ticket-triage effort.
- Improve customer-support efficiency and response consistency.

## Problem Statements

### PS-1: Ticket Intent Classification

Classify customer queries into five categories:

- Billing Issue
- Technical Issue
- Account Management
- Complaint
- General Inquiry

Approach:
- Filter inbound customer messages.
- Clean and preprocess text.
- Generate intent labels using rule-based heuristics.
- Convert text into TF-IDF features.
- Train and compare classification models.
- Evaluate using Accuracy, Precision, Recall and F1-score.

---

### PS-2: Ticket Priority / Urgency Detection

Classify customer queries into three priority levels:

- Low
- Medium
- High

Approach:
- Generate priority labels using urgency-related keywords and text indicators.
- Use TF-IDF features.
- Train classification models.
- Compare Logistic Regression and Linear SVM.
- Tune the Linear SVM hyperparameter C.
- Evaluate using Accuracy, Macro Precision, Macro Recall, Macro F1 and High Priority Recall.

Final model:
- Linear SVM
- C = 0.1
- class_weight = balanced

---

### PS-3: Automated Response Generation

Generate a suitable first-draft response for a customer query.

Approach:
- Reconstruct customer-query and company-response pairs.
- Clean and preprocess both queries and responses.
- Tokenize the text.
- Apply padding and truncation.
- Build a Sequence-to-Sequence model.
- Use LSTM Encoder and Decoder.
- Use Attention mechanism.
- Apply teacher forcing during training.
- Generate responses token by token.

Response reliability is improved using:
- Response quality checking.
- TF-IDF retrieval fallback.
- Cosine similarity.
- Safe fallback response.

Architecture:

Customer Query
      ↓
Text Preprocessing
      ↓
Tokenization
      ↓
Encoder LSTM
      ↓
Attention
      ↓
Decoder LSTM
      ↓
Generated Response
      ↓
Quality Check
      ↓
TF-IDF Retrieval Fallback
      ↓
Final Response

---

### PS-4: Response Time Prediction

Predict how long a company is likely to take to respond to a customer query.

This is a supervised regression problem.

Response time is calculated using:

Response Time = Company Reply Timestamp - Customer Query Timestamp

Features include:

- Message length
- Word count
- Hour of day
- Day of week
- Urgency indicators
- Predicted intent
- Predicted priority
- Responding company/handle

Regression models:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting / XGBoost Regressor

Evaluation metrics:

- MAE
- RMSE
- R² Score
- Feature Importance

---

## End-to-End Workflow

Customer Query
      ↓
PS-1: Intent Detection
      ↓
PS-2: Priority Detection
      ↓
PS-3: Response Generation
      ↓
PS-4: Response Time Prediction
      ↓
Intelligent Customer Support Ticket

---

## Dataset

The project uses the Customer Support on Twitter (TWCS) dataset.

The dataset contains customer-support conversations between customers and major companies.

Important columns include:

- tweet_id
- author_id
- inbound
- created_at
- text
- response_tweet_id
- in_response_to_tweet_id

The threading columns are used to reconstruct customer-company conversations.

---

## Data Preprocessing

The project performs the following preprocessing operations:

- Lowercase conversion
- URL removal
- @mention removal
- Special-character handling
- Whitespace normalization
- Duplicate removal
- Empty-text removal
- Tokenization
- Padding and truncation

---

## Technologies Used

### Programming
- Python

### Data Processing
- Pandas
- NumPy

### NLP
- NLTK
- spaCy
- TF-IDF
- N-grams

### Machine Learning
- Scikit-learn
- Logistic Regression
- Linear SVM
- Random Forest
- Linear Regression
- Gradient Boosting
- XGBoost

### Deep Learning
- TensorFlow
- Keras
- LSTM
- RNN
- Seq2Seq
- Attention

### Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- Macro F1
- MAE
- RMSE
- R²
- BLEU
- ROUGE

---

## Project Structure

Intelligent-Customer-Support-Ticket-System/
│
├── data/
│   └── twcs.csv
│
├── notebooks/
│   ├── PS1_Intent_Classification.ipynb
│   ├── PS2_Priority_Detection.ipynb
│   ├── PS3_Response_Generation.ipynb
│   └── PS4_Response_Time_Prediction.ipynb
│
├── models/
│   ├── PS1/
│   ├── PS2/
│   ├── PS3/
│   └── PS4/
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
├── README.md
└── .gitignore

---

## PS-3 Saved Models

The response-generation component saves:

- ps3_encoder.keras
- ps3_decoder.keras
- ps3_seq2seq_attention.keras
- ps3_query_tokenizer.pkl
- ps3_response_tokenizer.pkl
- ps3_retrieval_bundle.pkl

---

## Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd Intelligent-Customer-Support-Ticket-System
