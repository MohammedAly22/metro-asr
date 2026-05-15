import sys
sys.path.insert(0, '.')

from datasets import load_from_disk
from metro_asr.model.tokenizer import BPETokenizer

texts = load_from_disk('data_prepared/train').remove_columns(['audio'])['text']
BPETokenizer.train(texts[:200000], 'tokenizer/bpe', vocab_size=2000)
print('Done')