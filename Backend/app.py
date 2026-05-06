# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import torch
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Terminal UI
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.prompt import Prompt
from rich.table import Table

# Setup
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
console = Console()

# -----------------------------
# 1) Medical Knowledge Base
# -----------------------------
knowledge_base = [
    "Important Notice: This medical chatbot assistant is intended for informational and educational purposes only. It is not a substitute for evaluation by a licensed healthcare professional. ",
    "If you are experiencing severe symptoms, your symptoms are getting worse, or you think you may have a medical emergency, seek immediate medical attention or call your local emergency number right away. ",
    "Pain is a signal from your nervous system that something may be wrong. It is an unpleasant feeling, such as a prick, tingle, sting, burn, or ache. Pain may be sharp or dull. You may feel pain in one area of your body or all over. ",
    "Each person feels pain differently, even if the reason for the pain is the same. ",
    "Chronic pain can occur anywhere in your body and may cause other symptoms such as fatigue, mood changes or difficulty sleeping. ",
    "Many older adults have chronic pain. ",
    "Chronic pain lasts three months or longer or when pain continues after your body has healed. If the cause of your pain is unknown, your health care provider may ask you about your medical history, describe the pain and how it affects your life, do a physical exam, order blood tests or other medical tests. ",
    "Chronic Pain is not always curable, but treatments can help. Treatments may include medicines, including pain relievers. There are also non-drug treatments, such as acupuncture, physical therapy and sometimes surgery. ",
    "Stress is how your brain and body response to a challenge or demand. ",
    "When you are stressed, your body releases certain hormones. ",
    "Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. ",
    "Bronchitis causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tightness. ",
    "There are two main types of bronchitis: acute and chronic. ",
    "To diagnose acute bronchitis, your health care provider will ask about your symptoms and listen to your breathing. You may also have other tests. ",
    "Chest pain due to a heart condition may include pressure, tightness, squeezing or aching in the chest, shortness of breath, fatigue, cold sweats, fast heartbeat . Pain that spreads to the shoulder, arm, back, neck, jaw, teeth or upper belly. ",
    "Chest pain not likely due to a heart condition includes a sour taste or feeling of food coming back in the mouth, pain that gets worse when breathing deeply or coughing, tenderness when pushing on chest ",
    "Symptoms of constipation may include fewer than 3 bowel movements a week, stools that are hard, dry or lumpy, stools that are difficult or painful to pass, a feeling that not all stool has passed, see a doctor if the symptoms do not go away or there is family history of colon or rectal cancer ",
    "If a cough lasts less than 3 weeks it is called acute cough and infections or flares of chronic lung conditions cause acute coughs, if a cough lasts longer than eight weeks in adults and 4 weeks in children it is called chronic and it is related to underlying lung, heart or sinus condition ",
    "Coughing is a reflex that clears the air ways of mucus, irritants, or foreign particles. It can be dry(no mucus) or productive (with mucus or phlegm) and maybe triggered by infections,allergies, or irritants like smoke ",
    "Diarrhea is loose, watery stools. You have diarrhea if you have loose stools three or more times in one day, diarrhea lasting more than a few days might be a symptom of a chronic disease ",
    "Body temperature higher than 100F (37.8 C) is considered to be a fever, other fever signs may include sweating, chills and shivering, headache, muscle aches, loss of appetite, irritability, dehydration, general weakness. ",
    "Frostbite is an injury caused by freezing of skin and underlying tissues. Symptoms of frostbite include numbness, tingling, patches of skin in red, white, blue, gray, purple or brown. Cold, hard, waxy looking skin. Clumsiness due to joint stiffness. Pain. Blistering after rewarming ",
    "Common Symptoms related to gas in the digestive tract include belching, bloating and distention, and passing gas, Gas enters digestive tract when bacteria in large intestine breaks down certain undigested carbohydrates ",
    "Any type of transportation might cause motion sickness. Feeling ill while in motion, symptoms may advance from general feeling of discomfort to a cold sweat, dizziness and vomiting ",
    "Feeling like not getting enough air to breathe is called shortness of breath also called dyspnea. Symptoms include tightness of chest, trouble breathing or can’t breathe at all, suffocating ",
    "Absence seizures involve brief, sudden lapses of consciousness ",
    "Achilles tendinitis is an injury of the Achilles tendon It can be caused by using too much or too hard the tendon without enough rest ",
    "Acute liver failure is a loss of liver function that happens quickly usually in a person who has no preexisting liver disease ",
    "Anemia is a problem of not having enough healthy red blood cells or hemoglobin to carry oxygen to the body’s tissues ",
    "Aneurysm is a bulge or ballooning in the wall of the blood vessel ",
    "Angina is a type of chest pain caused by reduced blood flow to the heart. It is a symptom of coronary artery disease ",
    "Aphasia is a disorder that affects how you communicate which impact the speech as well the way you write and understand the spoken and written language ",
    "Berger disease is a kidney disease. It happens when a germ-fighting protein (immunoglobulin A) builds up in the kidneys ",
    "Dehydration occurs when the body uses or loses more fluid than it takes in. Then the body does not have enough water and other fluids to do its usual work ",
    "Nasal congestion is a blocked or stuffy feeling in the nose that makes breathing through it difficult. It is often due to swollen nasal tissues from colds,allergies or sinus infections ",
    "Depression is a mood disorder that causes a persistent feeling of sadness and loss of interest "
]

# -----------------------------
# 2) HARDENED CSV Logging
# -----------------------------
# We use an absolute path to ensure the file is created in the script's directory
CSV_PATH = os.path.join(os.getcwd(), "web_scrape_log.csv")

def log_to_csv(query, source, content):
    """Logs the exact data retrieved to a CSV file for testing purposes."""
    fieldnames = ["Timestamp", "User_Query", "Source", "Retrieved_Content"]
    file_exists = os.path.isfile(CSV_PATH)
    
    try:
        with open(CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "User_Query": query,
                "Source": source,
                "Retrieved_Content": str(content).replace('\n', ' ')
            })
    except Exception as e:
        # Fallback print if file writing fails
        print(f"DEBUG: CSV Log Error: {e}")

# -----------------------------
# 3) FIXED Web Scraper (NIH Search Engine)
# -----------------------------
def scrape_medical_info(query):
    """
    Scrapes the NIH/MedlinePlus Vivisimo search engine specifically.
    """
    # The specific URL provided by the user
    base_url = "https://vsearch.nlm.nih.gov/vivisimo/cgi-bin/query-meta"
    params = {
        "v:project": "medlineplus",
        "v:sources": "medlineplus-bundle",
        "query": query
    }
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(base_url, params=params, headers=headers, timeout=15)
        
        if response.status_code != 200:
            log_to_csv(query, f"HTTP FAIL {response.status_code}", "No connection")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # In the Vivisimo engine, search results are often in spans with class 'snippet'
        # or divs with class 'document-snippet'
        snippets = soup.find_all(['span', 'div'], class_=['snippet', 'document-snippet', 'description'])
        
        if not snippets:
            # Fallback: find any text within the results-list container
            results_container = soup.find(id='results-list')
            if results_container:
                snippets = results_container.find_all('p')

        if snippets:
            # Combine the first 3 snippets found
            extracted_text = " ".join([s.get_text(strip=True) for s in snippets[:3]])
            if len(extracted_text) > 10:
                log_to_csv(query, "SUCCESS (MedlinePlus)", extracted_text)
                return extracted_text
            
        log_to_csv(query, "FAILED (No Content Found)", "The page loaded but no snippets were detected.")
        return None
        
    except Exception as e:
        log_to_csv(query, f"CRITICAL ERROR: {str(e)}", "N/A")
        return None

# -----------------------------
# 4) AI Model Setup
# -----------------------------
console.print("[bold cyan]🤖 Initializing AI Models and Knowledge Base...[/bold cyan]")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

kb_vectors = embedder.encode(knowledge_base)
kb_vectors = kb_vectors / np.linalg.norm(kb_vectors, axis=1, keepdims=True)

# -----------------------------
# 5) UI Layout and Main Logic
# -----------------------------
DISCLAIMER = "Educational purposes only. Seek professional medical help for emergencies."

def get_layout():
    l = Layout()
    l.split(Layout(name="header", size=3), Layout(name="body"), Layout(name="footer", size=3))
    l["body"].split_row(Layout(name="side", ratio=1), Layout(name="main", ratio=3))
    return l

layout = get_layout()
layout["header"].update(Panel("🏥 MEDICAL INFORMATION ASSISTANT", style="bold white on blue"))
layout["side"].update(Panel("Local Topics:\n• Fever\n• Chronic Pain\n• Heart\n• Bronchitis", title="System Knowledge", border_style="blue"))
layout["footer"].update(Panel(f"Test Log: {CSV_PATH} | Type 'exit' to quit", style="italic grey50"))

history = []

# Start session with explicit disclaimer
console.clear()
console.print(Panel(f"[bold red]MEDICAL DISCLAIMER[/bold red]\n{DISCLAIMER}", border_style="red"))

while True:
    chat_render = Text()
    for q, r in history[-2:]:
        chat_render.append(f"User: {q}\n", style="bold yellow")
        chat_render.append(Text.from_markup(f"{r}\n\n"))
    
    layout["main"].update(Panel(chat_render, title="Chat History", border_style="green"))

    with Live(layout, refresh_per_second=2, screen=False):
        pass
    
    user_input = Prompt.ask("\n[bold white]Question[/bold white]")
    if user_input.lower() in ['exit', 'quit']: 
        break

    # 1. Similarity Logic
    q_vec = embedder.encode([user_input])
    q_vec = q_vec / np.linalg.norm(q_vec)
    scores = (q_vec @ kb_vectors.T).flatten()
    top_score = scores[np.argmax(scores)]
    
    context = ""
    source = ""

    # 2. Source Selection (Threshold 0.45)
    if top_score < 0.45:
        with console.status("[bold magenta]Searching NIH MedlinePlus Online Database...[/bold magenta]"):
            web_info = scrape_medical_info(user_input)
            if web_info:
                context = web_info
                source = "Web Search (MedlinePlus)"
            else:
                context = "No specific information found in the local database or MedlinePlus."
                source = "No Data Found"
    else:
        context = knowledge_base[np.argmax(scores)]
        source = "Local Knowledge Base"

    # 3. AI Answer Generation (Fixing the "Iii." issue)
    # We provide a very structured prompt to avoid hallucinated garbage.
    prompt = (
        f"Background Medical Information: {context}\n\n"
        f"User Question: {user_input}\n\n"
        "Instructions: Based on the background info above, write a clear 1-2 sentence explanation. "
        "If the background information is empty or irrelevant, say 'I do not have enough info.' "
        "Do not provide single-word answers or gibberish."
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            num_beams=2, 
            early_stopping=True
        )
    ai_msg = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Final formatting
    formatted_res = f"AI: {ai_msg}\n[dim]Source: {source}[/dim]\n[dim]Note: {DISCLAIMER}[/dim]"
    history.append((user_input, formatted_res))