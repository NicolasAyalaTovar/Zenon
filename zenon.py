"""Zenon: a Stoic-philosophy chatbot built on a fine-tuned GPT-2.

Usage:
    python zenon.py train      # fine-tune GPT-2 on the quotes in data/
    python zenon.py chat       # open the desktop chat interface
"""

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext

import torch
from PIL import Image, ImageTk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
AVATAR_PATH = BASE_DIR / "assets" / "avatar.jpg"
MODEL_DIR = BASE_DIR / "fitted_model"

FONT = ("Times New Roman", 12)
BACKGROUND_COLOR = "#eaeaea"
TEXT_COLOR = "#4f6d7a"


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def load_texts(data_dir: Path) -> list[str]:
    """Read every non-empty line from the .txt files in ``data_dir``."""
    texts = []
    for path in sorted(data_dir.glob("*.txt")):
        texts.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return texts


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length: int = 512):
        self.encodings = [
            tokenizer.encode(text + tokenizer.eos_token, max_length=max_length, truncation=True)
            for text in texts
        ]

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx], dtype=torch.long)


def make_collate(pad_token_id: int):
    """Pad a batch of variable-length sequences; padded positions are ignored in the loss."""
    def collate(batch):
        input_ids = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)
        attention_mask = pad_sequence([torch.ones_like(x) for x in batch], batch_first=True, padding_value=0)
        labels = input_ids.masked_fill(attention_mask == 0, -100)
        return input_ids, attention_mask, labels
    return collate


def train(epochs: int = 4, batch_size: int = 2, lr: float = 5e-5, base_model: str = "gpt2") -> None:
    texts = load_texts(DATA_DIR)
    print(f"Loaded {len(texts)} quotes from {DATA_DIR}")

    tokenizer = GPT2Tokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size,
                            collate_fn=make_collate(tokenizer.pad_token_id))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=10, num_training_steps=len(dataloader) * epochs)

    for epoch in range(epochs):
        total = 0.0
        for input_ids, attention_mask, labels in dataloader:
            outputs = model(input_ids.to(device), attention_mask=attention_mask.to(device),
                            labels=labels.to(device))
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += outputs.loss.item()
        print(f"Epoch {epoch + 1}/{epochs} - mean loss: {total / len(dataloader):.4f}")

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")


# -----------------------------------------------------------------------------
# Chat interface
# -----------------------------------------------------------------------------

class Zenon:
    """Wraps the fine-tuned model so it is loaded only once."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        if not model_dir.exists():
            raise SystemExit(f"No trained model found in {model_dir}. Run `python zenon.py train` first.")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir).eval()

    @torch.no_grad()
    def answer(self, question: str, max_new_tokens: int = 60) -> str:
        inputs = self.tokenizer(question, return_tensors="pt")
        output = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.92,
            temperature=0.8, no_repeat_ngram_size=3, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


class ChatbotGUI:
    def __init__(self, master: tk.Tk, bot: Zenon):
        self.bot = bot
        master.title("Zenon")
        master.configure(bg=BACKGROUND_COLOR)

        image = Image.open(AVATAR_PATH).resize((300, 300), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        avatar = tk.Label(master, image=photo, bg=BACKGROUND_COLOR)
        avatar.image = photo
        avatar.pack(side=tk.TOP)

        tk.Label(master, text="How can I help you?", font=FONT,
                 bg=BACKGROUND_COLOR, fg=TEXT_COLOR).pack()

        self.entry = tk.Entry(master, font=FONT, bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
                              insertbackground=TEXT_COLOR, width=40)
        self.entry.pack()
        self.entry.bind("<Return>", lambda _event: self.process_question())

        self._button(master, "Ask", self.process_question).pack(pady=(5, 10))

        self.response_area = scrolledtext.ScrolledText(master, height=10, width=50, font=FONT,
                                                       bg=BACKGROUND_COLOR, fg=TEXT_COLOR, wrap=tk.WORD)
        self.response_area.pack(padx=10)

        self._button(master, "Exit", master.quit).pack(pady=(5, 10))

    @staticmethod
    def _button(master, text, command):
        return tk.Button(master, text=text, font=FONT, bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
                         activebackground=BACKGROUND_COLOR, command=command, width=8, height=1)

    def process_question(self) -> None:
        question = self.entry.get().strip()
        if not question:
            return
        self.response_area.delete(1.0, tk.END)
        self.response_area.insert(tk.END, self.bot.answer(question))


def main() -> None:
    parser = argparse.ArgumentParser(description="Zenon, a Stoic-philosophy chatbot.")
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train", help="Fine-tune GPT-2 on the quotes in data/.")
    train_parser.add_argument("--epochs", type=int, default=4)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--lr", type=float, default=5e-5)
    sub.add_parser("chat", help="Open the chat interface.")
    args = parser.parse_args()

    if args.command == "train":
        train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    else:
        bot = Zenon()
        root = tk.Tk()
        ChatbotGUI(root, bot)
        root.mainloop()


if __name__ == "__main__":
    main()
