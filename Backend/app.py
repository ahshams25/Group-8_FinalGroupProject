import customtkinter as ctk
import threading
import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- 1. MODEL & KNOWLEDGE BASE SETUP ---
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

DISCLAIMER = "\n\n(Note: Seek professional help.)"
CONF_THRESHOLD = 0.22

# Load Models
print("Loading AI Models... please wait.")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
kb_vectors = embedder.encode(knowledge_base, convert_to_numpy=True)
kb_vectors = kb_vectors / (np.linalg.norm(kb_vectors, axis=1, keepdims=True) + 1e-12)

MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# --- 2. UI IMPLEMENTATION ---

class MedicalChatbotApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Config
        self.title("Medical Info AI Assistant")
        self.geometry("900x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Layout Grid
        self.grid_rowconfigure(0, weight=1)  
        self.grid_columnconfigure(1, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        
        self.sidebar_label = ctk.CTkLabel(self.sidebar, text="Medical AI", font=ctk.CTkFont(size=20, weight="bold"))
        self.sidebar_label.grid(row=0, column=0, padx=20, pady=20)
        
        self.info_label = ctk.CTkLabel(self.sidebar, text="This tool provides info\nbased on a medical DB.\n\nNot for diagnosis.", 
                                      wraplength=160, font=ctk.CTkFont(size=12))
        self.info_label.grid(row=1, column=0, padx=20, pady=10)

        # --- Chat Display Area (The "ChatGPT" Box) ---
        self.chat_display = ctk.CTkTextbox(self, state="disabled", wrap="word", font=("Arial", 14))
        self.chat_display.grid(row=0, column=1, padx=20, pady=(20, 0), sticky="nsew")

        # --- Input Frame ---
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=1, column=1, padx=20, pady=20, sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.user_entry = ctk.CTkEntry(self.input_frame, placeholder_text="Ask about symptoms (e.g. fever, bronchitis)...", height=40)
        self.user_entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.user_entry.bind("<Return>", lambda event: self.send_message()) 

        self.send_button = ctk.CTkButton(self.input_frame, text="Send", width=100, height=40, command=self.send_message)
        self.send_button.grid(row=0, column=1)

        # Initial Greeting
        self.append_chat("Bot", "Hello! I am your medical information assistant. How can I help you today?")

    def append_chat(self, sender, message):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"{sender}: {message}\n\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    def send_message(self):
        user_text = self.user_entry.get().strip()
        if not user_text:
            return

        self.append_chat("You", user_text)
        self.user_entry.delete(0, "end")
        
        # Run AI logic in a separate thread to keep UI smooth
        thread = threading.Thread(target=self.generate_bot_response, args=(user_text,))
        thread.daemon = True
        thread.start()

    def generate_bot_response(self, question):
        q_clean = question.lower()

        # Logic for greetings and small talk
        if q_clean in ["hi", "hello", "hey", "hi there"]:
            response = "Hello! What symptoms or medical topics are you curious about today?"
        elif len(set(q_clean)) < 4:
            response = "I'm sorry, I didn't quite understand that. Please ask a health-related question."
        else:
            # Vector Search / Agentic Retrieval
            q_vec = embedder.encode([question], convert_to_numpy=True)
            q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12)
            scores = (q_vec @ kb_vectors.T).flatten()
            
            top_idx = np.argmax(scores)
            
            if scores[top_idx] < CONF_THRESHOLD:
                response = "I don't have enough information in my database to answer that specifically."
            else:
                context = knowledge_base[top_idx]
                prompt = (f"Context: {context}\n"
                          f"Question: {question}\n"
                          f"Answer the question using only the context provided in a full sentence:")
                
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=100)
                response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().capitalize()

        # Use .after() to safely update UI from a different thread
        self.after(0, lambda: self.append_chat("Bot", f"{response}{DISCLAIMER}"))

if __name__ == "__main__":
    app = MedicalChatbotApp()
    app.mainloop()