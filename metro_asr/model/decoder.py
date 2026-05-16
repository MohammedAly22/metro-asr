"""
CTC Beam Search Decoder with optional KenLM language model.

Greedy decoding is fast but makes independent per-frame decisions.
Beam search + LM rescoring produces valid words by leveraging language context.
"""
import torch
import numpy as np


class CTCBeamSearchDecoder:
    def __init__(self, tokenizer, lm_path=None, beam_width=20, alpha=0.5, beta=1.0):
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.is_bpe = hasattr(tokenizer, '_sp') and tokenizer._sp is not None

        vocab = self._build_vocab()

        if lm_path:
            from pyctcdecode import build_ctcdecoder
            import kenlm
            self.decoder = build_ctcdecoder(
                labels=vocab,
                kenlm_model_path=lm_path,
                alpha=alpha,
                beta=beta,
            )
            self.has_lm = True
        else:
            from pyctcdecode import build_ctcdecoder
            self.decoder = build_ctcdecoder(labels=vocab)
            self.has_lm = False

    def _build_vocab(self):
        vocab = []
        seen = set()
        for i in range(self.tokenizer.vocab_size):
            if i == self.tokenizer.blank_id:
                token = ""
            elif i == self.tokenizer.unk_id:
                token = "⁇"
            elif i == self.tokenizer.pad_id:
                token = "⁈"
            elif hasattr(self.tokenizer, 'id_to_token'):
                token = self.tokenizer.id_to_token.get(i, "")
            elif hasattr(self.tokenizer, '_sp') and self.tokenizer._sp is not None:
                sp_id = i - 3
                if 0 <= sp_id < self.tokenizer._sp.GetPieceSize():
                    token = self.tokenizer._sp.IdToPiece(sp_id)
                else:
                    token = ""
            else:
                token = ""

            if token in seen and token != "":
                token = f"<dup_{i}>"
            seen.add(token)
            vocab.append(token)
        return vocab

    def decode(self, log_probs, lengths=None):
        results = []
        log_probs_np = log_probs.float().cpu().numpy()

        for i in range(log_probs_np.shape[0]):
            max_t = lengths[i].item() if lengths is not None else log_probs_np.shape[1]
            lp = log_probs_np[i, :max_t, :]

            text = self.decoder.decode(
                lp,
                beam_width=self.beam_width,
            )
            results.append(text.strip())

        return results

    def decode_batch(self, log_probs, lengths=None, batch_size=16):
        log_probs_np = log_probs.float().cpu().numpy()

        if lengths is not None:
            pool_input = []
            for i in range(log_probs_np.shape[0]):
                max_t = lengths[i].item()
                pool_input.append(log_probs_np[i, :max_t, :])
        else:
            pool_input = [log_probs_np[i] for i in range(log_probs_np.shape[0])]

        with self.decoder.pool(processes=min(4, len(pool_input))) as pool:
            results = pool.decode_batch(
                pool_input,
                beam_width=self.beam_width,
            )

        return [r.strip() for r in results]


def build_decoder(tokenizer, lm_path=None, beam_width=20, alpha=0.5, beta=1.0):
    if lm_path:
        return CTCBeamSearchDecoder(
            tokenizer, lm_path=lm_path,
            beam_width=beam_width, alpha=alpha, beta=beta,
        )
    return None
