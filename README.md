
---

#  Spam-Ham Classifier

##  Project Description

This project presents a simple and interactive web application that classifies email messages as either **"Spam"** or **"Ham"** (not spam). Built using **Streamlit**, it utilizes a **pre-trained machine learning model** for real-time classification based on user input.

---

##  Features

* **User-Friendly Interface**: Clean, intuitive web UI powered by Streamlit.
* **Real-time Classification**: Instantly classifies entered messages.
* **Pre-trained Model**: Ensures quick and accurate predictions.
* **Robust Text Preprocessing**: Includes:

  * Lowercasing
  * Punctuation removal
  * Stopword removal
  * Word stemming

---

##  How It Works

1. **Preprocessing**:

   * Convert input text to lowercase.
   * Remove punctuation.
   * Apply stemming (e.g., "running" → "run").
   * Remove stopwords (e.g., "the", "is").

2. **Vectorization**:

   * Text is transformed into numerical format using `CountVectorizer`.

3. **Prediction**:

   * The `RandomForestClassifier` model processes the vectorized input.

4. **Result Display**:

   * The app outputs either **"SPAM"** or **"HAM"**.

---

##  Files in this Repository

| File/Folder           | Description                                    |
| --------------------- | ---------------------------------------------- |
| `app.py`              | Streamlit app script                           |
| `spam_detector.py`    | Model training script (generates `.pkl` files) |
| `spam_classifier.pkl` | Pre-trained RandomForestClassifier model       |
| `vectorizer.pkl`      | Pre-trained CountVectorizer                    |
| `requirements.txt`    | Required Python packages                       |
| `.gitignore`          | Files/folders excluded from Git versioning     |

---

##  Dataset

The model was trained using the **`spam_ham_dataset.csv`** dataset.

 

➡ **Download here**:
(https://drive.google.com/file/d/1kXrMjjIfYXieD_fw5SvXi1KXKhMYxmxl/view)

After downloading, place the file in the project root folder:

```
/Spam_Detection/spam_ham_dataset.csv
```

---

##  Local Setup & Installation

### 1. Clone the repository:

```bash
git clone https://github.com/dkewat25/Spam_Detection.git
cd Spam_Detection
```

### 2. Download the dataset:

Place `spam_ham_dataset.csv` in the root directory.

### 3. Create a virtual environment (recommended):

```bash
python -m venv venv
```

### 4. Activate the virtual environment:

* **Windows**:

  ```bash
  .\venv\Scripts\activate
  ```

* **macOS/Linux**:

  ```bash
  source venv/bin/activate
  ```

### 5. Install dependencies:

```bash
pip install -r requirements.txt
```

### 6. (Optional) Retrain the model:

```bash
python spam_detector.py
```

### 7. Run the Streamlit app:

```bash
streamlit run app.py
```

The app will launch in your browser.

---

##   Usage

1. **Enter Email Text**: Paste/type an email into the textbox.
2. **Click "🔍 Classify"**: Let the model analyze your input.
3. **View Result**: The prediction (`SPAM` or `HAM`) will be shown.

---

##  Live Demo

Access the deployed app here:
[🔗 Streamlit App Link](https://spamclassifierdk.streamlit.app/)
*

---

