"""
SentencePiece BPE tokenizer.

CTC reserves index 0 for the blank symbol, so every SentencePiece id is shifted
by ``N_SPECIAL`` when encoding and shifted back when decoding:

    id 0 -> <blank>   (CTC blank; never emitted by the tokenizer)
    id 1 -> <unk>
    id 2 -> <pad>
    id 3 -> SentencePiece piece 0
    ...
"""

import os

N_SPECIAL = 3
SPECIAL_TOKENS = ["<blank>", "<unk>", "<pad>"]


class BPETokenizer:
    def __init__(self, model_path=None):
        self._sp = None
        self._model_path = model_path
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
        return self._sp.GetPieceSize() + N_SPECIAL

    def encode(self, text):
        text = text.lower()
        return [i + N_SPECIAL for i in self._sp.EncodeAsIds(text)]

    def decode(self, ids):
        adjusted = [i - N_SPECIAL for i in ids if i >= N_SPECIAL]
        return self._sp.DecodeIds(adjusted)

    def id_to_piece(self, token_id):
        """Raw SentencePiece piece for a model id, or '' for a special token."""
        sp_id = token_id - N_SPECIAL
        if 0 <= sp_id < self._sp.GetPieceSize():
            return self._sp.IdToPiece(sp_id)
        return ""

    @classmethod
    def load(cls, model_path):
        return cls(model_path=model_path)


def build_tokenizer(config, tokenizer_dir="checkpoints"):
    tok_type = config["tokenizer"].get("type", "bpe")
    if tok_type != "bpe":
        raise ValueError(
            f"Unknown tokenizer type: {tok_type!r}. Metro-ASR only ships a BPE tokenizer."
        )

    model_path = os.path.join(tokenizer_dir, "bpe.model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"BPE model not found at {model_path}. Download a released checkpoint, or "
            "train one with: python scripts/train_bpe_tokenizer.py"
        )

    tokenizer = BPETokenizer.load(model_path)
    config["tokenizer"]["vocab_size"] = tokenizer.vocab_size
    return tokenizer
