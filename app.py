import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from transformers import MarianMTModel, MarianTokenizer  # type: ignore

class TranslationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Translation App")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Model cache
        self.models = {}
        self.tokenizers = {}
        
        # Language pairs mapping (model_name: (source, target))
        self.language_pairs = {
            "English → Hindi": ("en", "hi"),
            "Hindi → English": ("hi", "en"),
            "English → Bangla": ("en", "bn"),
            "Bangla → English": ("bn", "en"),
            "Hindi → Bangla": ("hi", "bn"),
            "Bangla → Hindi": ("bn", "hi"),
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title_label = tk.Label(
            self.root, 
            text="Language Translation App", 
            font=("Arial", 18, "bold"),
            pady=10
        )
        title_label.pack()
        
        # Language selection frame
        lang_frame = tk.Frame(self.root)
        lang_frame.pack(pady=10)
        
        tk.Label(lang_frame, text="Translation Direction:", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
        
        self.lang_var = tk.StringVar(value="English → Hindi")
        lang_dropdown = ttk.Combobox(
            lang_frame, 
            textvariable=self.lang_var,
            values=list(self.language_pairs.keys()),
            state="readonly",
            width=25,
            font=("Arial", 10)
        )
        lang_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Input frame
        input_frame = tk.LabelFrame(self.root, text="Input Text", font=("Arial", 11, "bold"))
        input_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        self.input_text = scrolledtext.ScrolledText(
            input_frame, 
            wrap=tk.WORD, 
            font=("Arial", 11),
            height=8
        )
        self.input_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.translate_btn = tk.Button(
            button_frame,
            text="Translate",
            command=self.translate_text,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=5,
            cursor="hand2"
        )
        self.translate_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_text,
            bg="#f44336",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=5,
            cursor="hand2"
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Output frame
        output_frame = tk.LabelFrame(self.root, text="Translation", font=("Arial", 11, "bold"))
        output_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame, 
            wrap=tk.WORD, 
            font=("Arial", 11),
            height=8,
            state=tk.DISABLED
        )
        self.output_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_label = tk.Label(
            self.root, 
            text="Ready", 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            font=("Arial", 9)
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
    def get_model_name(self, src_lang, tgt_lang):
        """Get the appropriate MarianMT model name"""
        return f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    
    def load_model(self, model_name):
        """Load model and tokenizer (with caching)"""
        # Import transformers lazily so the module is only required when loading models,
        # and provide a clear message if the package is missing.
        try:
            from transformers import MarianMTModel, MarianTokenizer
        except Exception:
            messagebox.showerror(
                "Missing dependency",
                "The 'transformers' package is not installed or could not be imported.\n\n"
                "Install it with:\n\n    pip install transformers sentencepiece\n\n"
                "Then restart the application."
            )
            self.status_label.config(text="Missing 'transformers' package")
            raise

        if model_name not in self.models:
            self.status_label.config(text=f"Loading model: {model_name}...")
            self.root.update()
            
            try:
                self.tokenizers[model_name] = MarianTokenizer.from_pretrained(model_name)
                self.models[model_name] = MarianMTModel.from_pretrained(model_name)
                self.status_label.config(text="Model loaded successfully!")
            except Exception as e:
                raise Exception(f"Failed to load model: {str(e)}")
        
        return self.models[model_name], self.tokenizers[model_name]
    def perform_translation(self):
        """Perform the actual translation in a separate thread"""
        try:
            # Get selected language pair
            selected = self.lang_var.get()
            src_lang, tgt_lang = self.language_pairs[selected]

            # Get input text
            input_text = self.input_text.get("1.0", tk.END).strip()

            if not input_text:
                messagebox.showwarning("Warning", "Please enter text to translate!")
                self.translate_btn.config(state=tk.NORMAL)
                self.status_label.config(text="Ready")
                return

            # Try direct model first
            model_name = self.get_model_name(src_lang, tgt_lang)
            try:
                model, tokenizer = self.load_model(model_name)
                self.status_label.config(text="Translating (direct)...")
                self.root.update()

                inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
                translated = model.generate(**inputs)
                translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

            except Exception as direct_err:
                # If direct model missing on HF, attempt chained translation via English when possible.
                err_msg = str(direct_err)
                # Check for the common HF "not a local folder and is not a valid model identifier" message
                if "not a local folder and is not a valid model identifier" in err_msg or "Failed to load model" in err_msg:
                    # Try a chained translation: src -> en -> tgt
                    if src_lang != "en" and tgt_lang != "en":
                        self.status_label.config(text="Direct model not found — attempting chained translation via English...")
                        self.root.update()

                        # Load src -> en
                        try:
                            model1, tokenizer1 = self.load_model(self.get_model_name(src_lang, "en"))
                        except Exception as e1:
                            raise Exception(f"Could not load intermediate model {src_lang}->en: {e1}")

                        # Load en -> tgt
                        try:
                            model2, tokenizer2 = self.load_model(self.get_model_name("en", tgt_lang))
                        except Exception as e2:
                            raise Exception(f"Could not load intermediate model en->{tgt_lang}: {e2}")

                        # First step: src -> en
                        inputs1 = tokenizer1(input_text, return_tensors="pt", padding=True, truncation=True)
                        inter = model1.generate(**inputs1)
                        inter_text = tokenizer1.decode(inter[0], skip_special_tokens=True)

                        # Second step: en -> tgt
                        inputs2 = tokenizer2(inter_text, return_tensors="pt", padding=True, truncation=True)
                        translated = model2.generate(**inputs2)
                        translated_text = tokenizer2.decode(translated[0], skip_special_tokens=True)

                    else:
                        # No sensible chained path, re-raise original error
                        raise
                else:
                    # Unexpected error — re-raise
                    raise

            # Display result
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert("1.0", translated_text)
            self.output_text.config(state=tk.DISABLED)

            self.status_label.config(text="Translation complete!")

        except Exception as e:
            messagebox.showerror("Error", f"Translation failed:\n{str(e)}")
            self.status_label.config(text="Translation failed!")

        finally:
            self.translate_btn.config(state=tk.NORMAL)
    
    def translate_text(self):
        """Start translation in a separate thread"""
        self.translate_btn.config(state=tk.DISABLED)
        thread = threading.Thread(target=self.perform_translation, daemon=True)
        thread.start()
    
    def clear_text(self):
        """Clear all text fields"""
        self.input_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.status_label.config(text="Ready")

def main():
    root = tk.Tk()
    app = TranslationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()