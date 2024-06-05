import torch.nn as nn


def resize_token_embeddings(
    model,
    new_num_tokens
):

    old_embeddings = model.backbone.embeddings
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    new_embeddings = nn.Embedding(
        new_num_tokens,
        old_embedding_dim,
        device=old_embeddings.weight.device,
        dtype=old_embeddings.weight.dtype,
    )
    nn.init.normal_(new_embeddings.weight, std=0.02)
    n = min(old_num_tokens, new_num_tokens)
    new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
    model.backbone.embeddings = new_embeddings

    model.tie_weights()
