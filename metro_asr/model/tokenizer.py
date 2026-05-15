import json
import os

ARABIC_LETTERS = list(
    "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهويى"
)
ARABIC_DIACRITICS = list("َُِّْ")
ARABIC_EXTENDED_DIACRITICS = list("ًٌٍ")
ENGLISH_LOWER = list("abcdefghijklmnopqrstuvwxyz")
DIGITS = list("0123456789")
ARABIC_DIGITS = list("٠١٢٣٤٥٦٧٨٩")
PUNCTUATION = list(".,!?:;-'\"")
ARABIC_PUNCTUATION = list("،؛؟")

SPECIAL_TOKENS = ["<blank>", "<unk>", "<pad>"]


class CharTokenizer:
    def __init__(self, vocab=None):
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = self._build_default_vocab()
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.vocab)}
        self.blank_id = 0
        self.unk_id = 1
        self.pad_id = 2

    @staticmethod
    def _build_default_vocab():
        vocab = list(SPECIAL_TOKENS)
        vocab.append(" ")
        vocab.extend(ARABIC_LETTERS)
        vocab.extend(ARABIC_DIACRITICS)
        vocab.extend(ARABIC_EXTENDED_DIACRITICS)
        vocab.extend(ENGLISH_LOWER)
        vocab.extend(DIGITS)
        vocab.extend(ARABIC_DIGITS)
        vocab.extend(PUNCTUATION)
        vocab.extend(ARABIC_PUNCTUATION)
        return vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        text = text.lower()
        ids = []
        for ch in text:
            if ch in self.token_to_id:
                ids.append(self.token_to_id[ch])
            else:
                ids.append(self.unk_id)
        return ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            if i in self.id_to_token and i not in (self.blank_id, self.pad_id):
                tokens.append(self.id_to_token[i])
        return "".join(tokens)

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {"type": "char", "vocab": self.vocab}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab=data["vocab"])


class BPETokenizer:
    def __init__(self, model_path=None):
        self._sp = None
        self.blank_id = 0
        self.unk_id = 1
        self.pad_id = 2
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, model_path):
        import sentencepiece as spm
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)

    @property
    def vocab_size(self):
        if self._sp is None:
            raise RuntimeError("BPE model not loaded. Train or load a model first.")
        return self._sp.GetPieceSize() + 3

    def encode(self, text):
        text = text.lower()
        ids = self._sp.EncodeAsIds(text)
        return [i + 3 for i in ids]

    def decode(self, ids):
        adjusted = [i - 3 for i in ids if i >= 3]
        return self._sp.DecodeIds(adjusted)

    @staticmethod
    def train(texts, model_prefix, vocab_size=2000):
        import sentencepiece as spm
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            for text in texts:
                line = text.lower().strip()
                if line:
                    f.write(line + "\n")
            tmp_path = f.name

        spm.SentencePieceTrainer.Train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size - 3,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=-1,
            unk_id=0,
            bos_id=-1,
            eos_id=-1,
            max_sentence_length=16384,
            num_threads=4,
            train_extremely_large_corpus=False,
        )
        os.unlink(tmp_path)

    def save(self, path):
        data = {"type": "bpe", "model_path": self._model_path if hasattr(self, "_model_path") else ""}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, model_path):
        tokenizer = cls(model_path=model_path)
        tokenizer._model_path = model_path
        return tokenizer


def build_tokenizer(config, tokenizer_dir="tokenizer"):
    tok_cfg = config["tokenizer"]
    tok_type = tok_cfg.get("type", "char")

    if tok_type == "char":
        tokenizer_path = os.path.join(tokenizer_dir, "char_tokenizer.json")
        if os.path.exists(tokenizer_path):
            tokenizer = CharTokenizer.load(tokenizer_path)
        else:
            tokenizer = CharTokenizer()
            tokenizer.save(tokenizer_path)
        config["tokenizer"]["vocab_size"] = tokenizer.vocab_size
        return tokenizer

    if tok_type == "bpe":
        model_path = os.path.join(tokenizer_dir, "bpe.model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"BPE model not found at {model_path}. "
                "Run scripts/prepare_data.py first to train the tokenizer."
            )
        tokenizer = BPETokenizer.load(model_path)
        config["tokenizer"]["vocab_size"] = tokenizer.vocab_size
        return tokenizer

    raise ValueError(f"Unknown tokenizer type: {tok_type}")
