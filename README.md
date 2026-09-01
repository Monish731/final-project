# Intelligent Customer Support Ticket System

## NLP-Based Classification, Prioritization & Response Automation

An end-to-end Machine Learning and Natural Language Processing project that analyzes customer-support conversations and provides intelligent assistance for ticket routing, prioritization, response generation, and response-time prediction.

The project uses the Customer Support on Twitter (TWCS) dataset containing close to 3 million customer-support tweets exchanged with major brands.

---

## 📌 Project Overview

Customer-support teams receive a large number of messages every day. Manually understanding the customer's problem, determining its urgency, preparing a response, and estimating response time can be time-consuming.

This project develops an intelligent customer-support pipeline that addresses these tasks using four connected Problem Statements (PS):

1. Ticket Intent Classification
2. Ticket Priority / Urgency Detection
3. Automated Response Generation
4. Response Time Prediction

The overall system combines classical Machine Learning, NLP, Deep Learning, Sequence-to-Sequence modelling, and Regression.

---

# 🎯 Objectives

The main objectives of the project are:

- Automatically identify the intent of a customer query.
- Determine the urgency/priority of a customer ticket.
- Generate a suitable first-draft customer-support response.
- Predict the expected response time in minutes.
- Reduce manual ticket-triage effort.
- Support faster and more consistent customer service.
- Provide an end-to-end intelligent support workflow.

---

# 🧩 Problem Statements

## PS-1: Ticket Intent Classification

Classify incoming customer messages into five intent categories:

- Billing Issue
- Technical Issue
- Account Management
- Complaint
- General Inquiry

### Approach

- Filter inbound customer messages.
- Clean and preprocess text.
- Generate weak-supervision intent labels using rule-based keyword heuristics.
- Convert text into TF-IDF features.
- Train classification models.
- Compare model performance using Accuracy, Precision, Recall and F1-score.
- Select and save the best-performing model.

### Output

```text
Customer Query
      ↓
Intent Classifier
      ↓
Billing / Technical / Account /
Complaint / General
