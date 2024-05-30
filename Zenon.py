import tkinter as tk
from tkinter import scrolledtext, PhotoImage, font 
from PIL import Image, ImageTk
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, pipeline
from torch.utils.data import DataLoader, Dataset, RandomSampler
import os

def load_text(file_path):
    texts = []
    for file_path in file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            texts.extend(file.read().splitlines())
    return texts

file_path = [
    'C:/Users/EQUIPO/Documents/Portafolio/python/Zenon/t1.txt',
    'C:/Users/EQUIPO/Documents/Portafolio/python/Zenon/t2.txt',
    'C:/Users/EQUIPO/Documents/Portafolio/python/Zenon/t3.txt',
    'C:/Users/EQUIPO/Documents/Portafolio/python/Zenon/t4.txt'
]
stoic_texts = load_text(file_path)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = [tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True) for text in texts]

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx], dtype=torch.long)

def adjust_model(texts, model='gpt2', epochs=4):
    tokenizer = GPT2Tokenizer.from_pretrained(model)
    model = GPT2LMHeadModel.from_pretrained(model)
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = TextDataset(texts, tokenizer)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=2)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=len(dataloader) * epochs)

    for epoch in range(epochs):
        for batch in dataloader:
            inputs = batch.to(device)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    model.save_pretrained('./fitted_model')
    tokenizer.save_pretrained('./fitted_model')

times_font = ('Times New Roman', 12)  
background_color = '#eaeaea'  
text_color = '#4f6d7a'  
button_color = '#eaeaea'  
button_text_color = '#4f6d7a'  
button_active_color = '#eaeaea'

class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        master.title("Zenon")

        image_path = os.path.join('C:/Users/EQUIPO/Documents/Portafolio/python/Zenon/avatar.jpg')
        image = Image.open(image_path)
        image = image.resize((300, 300), Image.Resampling.LANCZOS) 
        photo = ImageTk.PhotoImage(image)

        self.image_label = tk.Label(master, image=photo)
        self.image_label.image = photo  
        self.image_label.pack(side=tk.TOP)

        self.label = tk.Label(master, text="How can I help you?", font=times_font, bg=background_color, fg=text_color)
        self.label.pack()

        self.entry = tk.Entry(master, font=times_font, bg=background_color, fg=text_color, insertbackground=text_color)
        self.entry.pack()

        self.submit_button = tk.Button(master, text="Ask", font=times_font, bg=button_color, fg=button_text_color, 
            activebackground=button_active_color, command=self.process_question, 
            width=8, height=1)
        self.submit_button.pack(pady=(5, 10))

        self.response_area = scrolledtext.ScrolledText(master, height=10, width=50, font=times_font, bg=background_color, fg=text_color)
        self.response_area.pack()

        self.quit_button = tk.Button(master, text="Out", font=times_font, bg=button_color, fg=button_text_color, 
            activebackground=button_active_color, command=master.quit, 
            width=8, height=1)
        self.quit_button.pack(pady=(5, 10))

    def process_question(self):
        question = self.entry.get()
        response = self.generate_response(question)
        self.show_response(response)

    def generate_response(self, question):
        model_path = './fitted_model'
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        chatbot = pipeline('text-generation', model=model, tokenizer=tokenizer)
        responses = chatbot(question, max_length=100, num_return_sequences=1)
        return responses[0]['generated_text']

    def show_response(self, response):
        self.response_area.delete(1.0, tk.END)
        self.response_area.insert(tk.END, response)

if __name__ == "__main__":
    root = tk.Tk()
    gui = ChatbotGUI(root)
    root.mainloop()
