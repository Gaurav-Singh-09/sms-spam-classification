# SMS Spam Classification using SimpleRNN

A deep learning Natural Language Processing (NLP) classification model built with TensorFlow and Keras to detect spam vs. legitimate (`ham`) SMS messages, achieving **98.45% test accuracy** on unseen data.

---

## Project Overview
- **Notebook:** `12oct_RNN_spam1_NN.ipynb`
- **Dataset:** `spam1.csv` — 6,776 labeled SMS messages (`ham` = `0`, `spam` = `1`) split 80/20 into training (5,420) and testing (1,356) sets.
- **Text Preprocessing:**
  - Removed redundant null columns (`Unnamed: 2`–`Unnamed: 4`) and encoded target labels.
  - Tokenized text vocabulary (`num_words=10000`) using Keras `Tokenizer`.
  - Applied sequence padding (`maxlen=50`) with `pad_sequences` for uniform input length.
  - One-hot encoded training labels via `to_categorical`.

---

## Model Architecture
Built using the Keras `Sequential` API:
1. **Embedding Layer:** `Embedding(input_dim=10000, output_dim=100)`
2. **Recurrent Layer:** `SimpleRNN(10)`
3. **Output Layer:** `Dense(2, activation='softmax')`
- **Compilation:** `Adam(learning_rate=0.01)` optimizer, `categorical_crossentropy` loss, trained for 5 epochs (`batch_size=63`, `validation_split=0.2`).

---

## Evaluation & Results
- **Validation Accuracy:** 98.34%
- **Test Accuracy:** **98.45%** (1,335 / 1,356 correct predictions)
- **Confusion Matrix (Test Set):**

| Actual \ Predicted | Ham (0) | Spam (1) |
| :--- | :---: | :---: |
| **Ham (0)** | 1,187 | 4 |
| **Spam (1)** | 17 | 148 |

---

## Tech Stack
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning & NLP:** Scikit-Learn, TensorFlow / Keras
- **Environment:** Jupyter Notebook

---

## How to Run
1. Clone this repository:
   ```bash
   git clone [https://github.com/Gaurav-Singh-09/sms-spam-classification.git](https://github.com/Gaurav-Singh-09/sms-spam-classification.git)
   cd sms-spam-classification
