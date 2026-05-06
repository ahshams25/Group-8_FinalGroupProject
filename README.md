# 🏥 Medical Information Assistant (AI Chatbot)

A terminal-based AI chatbot that provides medical information using a combination of:

* Local knowledge base
* Web scraping (NIH MedlinePlus)
* Transformer-based AI model (FLAN-T5)

> **Disclaimer**: This project is for educational purposes only and is not a substitute for professional medical advice.

# Features

* Interactive terminal chat interface using **Rich UI**
* Semantic search with **Sentence Transformers**
* Live medical data retrieval via **NIH MedlinePlus scraping**
* AI-generated responses using **Google FLAN-T5**
* CSV logging of web-scraped data for debugging/testing
* Intelligent fallback:

  * Uses local knowledge if relevant
  * Otherwise fetches data from the web

## Tech Stack

* Python
* PyTorch
* Transformers (Hugging Face)
* SentenceTransformers
* BeautifulSoup (Web Scraping)
* Requests
* Rich (Terminal UI)
* NumPy

## Project Structure

.
├── Backend/
│   └── app.py              # Main chatbot script
├── web_scrape_log.csv      # Auto-generated log file
└── README.md

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install torch transformers sentence-transformers beautifulsoup4 requests rich numpy
```

---

## How to Run

```bash
python Backend/app.py
```

---

## How It Works

### 1. User Input

* User enters a medical question in terminal

### 2. Semantic Matching

* Input is converted into vector embeddings
* Compared against local knowledge base

### 3. Decision Logic

* If similarity ≥ 0.45 → use local knowledge
* Else → scrape NIH MedlinePlus website

### 4. AI Response Generation

* Context + question passed into **FLAN-T5**
* Generates a clean 1–2 sentence response

### 5. Logging

* Web results are saved in:

```
web_scrape_log.csv
```

---

## Example

**Input:**

```
What is bronchitis?
```

**Output:**

```
AI: Bronchitis is inflammation of the airways that can cause coughing and mucus production.
Source: Local Knowledge Base
```

---

## Limitations

* Not a medical diagnosis tool
* Web scraping depends on NIH site structure
* Model responses may not always be perfect
* Requires internet for web fallback

---

## Future Improvements

* Add GUI (web or mobile)
* Improve response accuracy with fine-tuning
* Add voice input/output
* Use real medical APIs instead of scraping
* Store conversation history in database

---

## Authors

* Group 8

---

## License

This project is for academic use. Modify as needed.

---

## Medical Disclaimer

This chatbot is intended for **educational and informational purposes only**.
Always consult a licensed healthcare professional for medical concerns.

---
