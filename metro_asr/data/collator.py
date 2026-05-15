import torch


class MetroCollator:
    def __init__(self, pad_id=2):
        self.pad_id = pad_id

    def __call__(self, batch):
        features_list = [item["features"] for item in batch]
        tokens_list = [item["tokens"] for item in batch]
        texts = [item["text"] for item in batch]

        feature_lengths = torch.tensor([f.shape[0] for f in features_list], dtype=torch.long)
        token_lengths = torch.tensor([t.shape[0] for t in tokens_list], dtype=torch.long)

        max_feat_len = feature_lengths.max().item()
        max_tok_len = token_lengths.max().item()
        n_mels = features_list[0].shape[1]

        padded_features = torch.zeros(len(batch), max_feat_len, n_mels)
        padded_tokens = torch.full((len(batch), max_tok_len), self.pad_id, dtype=torch.long)

        for i, (feat, tok) in enumerate(zip(features_list, tokens_list)):
            padded_features[i, :feat.shape[0]] = feat
            padded_tokens[i, :tok.shape[0]] = tok

        return {
            "features": padded_features,
            "feature_lengths": feature_lengths,
            "tokens": padded_tokens,
            "token_lengths": token_lengths,
            "texts": texts,
        }
