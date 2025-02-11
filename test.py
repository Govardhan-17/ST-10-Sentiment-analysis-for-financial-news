import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt")  # Fixed download name
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class TextToNum:
    def __init__(self, text):
        self.text = text
        self.cleaned = ""
        self.tkns = []
        self.cl = []
        self.st = []

    def cleaner(self):
        text = re.sub(r',', '', self.text)
        cleaned_text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with one
        self.cleaned = cleaned_text.strip()  # Trim whitespace

    def token(self):
        if not self.cleaned:  # Ensure cleaner() is called first
            self.cleaner()
        self.tkns = word_tokenize(self.cleaned)

    def removeStop(self):
        if not self.tkns:  # Ensure tokenization happens first
            self.token()
        stop = set(stopwords.words('english'))  # Use set for faster lookup
        self.cl = [word.lower() for word in self.tkns if word.lower() not in stop]

    def stemme(self):
        if not self.cl:  # Ensure stopwords are removed first
            self.removeStop()
        ps = PorterStemmer()
        self.st = [ps.stem(word) for word in self.cl]
        return self.st

# Example Usage
text = "This is, an example sentence with punctuation!"
processor = TextToNum(text)

processor.cleaner()
processor.token()
processor.removeStop()
stemmed_output = processor.stemme()

print("Original Text:", text)
print("Cleaned Text:", processor.cleaned)
print("Tokens:", processor.tkns)
print("Without Stopwords:", processor.cl)
print("Stemmed Words:", stemmed_output)
