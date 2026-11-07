import torch
import re
from config import Config


def train_one_epoch(model, dataloader, criterion, optimizer, epoch, total_epochs=30):
    """
    Runs one full pass (epoch) over the training DataLoader using Teacher Forcing,
    computes cross-entropy loss, clips gradients, and updates model weights.
    """
    # Enable training mode (activates Dropout layers)
    model.train()
    total_loss = 0
    total_batches = len(dataloader)

    # Print an inline progress bar for visual feedback on CPU
    print(f"  Epoch {epoch:02d}/{total_epochs} Progress: [", end="", flush=True)
    dot_interval = max(1, total_batches // 12)

    for batch_idx, (src_batch, tgt_batch) in enumerate(dataloader):
        # Move input and target tensors onto the configured device (CPU)
        src_batch = src_batch.to(Config.DEVICE)
        tgt_batch = tgt_batch.to(Config.DEVICE)

        # Clear gradients accumulated from the previous batch
        optimizer.zero_grad()

        # Teacher Forcing shift:
        # tgt_input feeds [<sos>, tok1, tok2, ...] into the decoder
        # tgt_expected trains the model to predict [tok1, tok2, ..., <eos>]
        tgt_input = tgt_batch[:, :-1]
        tgt_expected = tgt_batch[:, 1:]

        # Forward pass through the 2-layer encoder and masked decoder
        output = model(src_batch, tgt_input)

        # Flatten batch and sequence dimensions to compute token-level loss
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt_expected.reshape(-1))

        # Backpropagate gradients through the network
        loss.backward()

        # Clip gradient norms to 1.0 to prevent exploding gradients during Transformer training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Apply weight updates via Adam optimizer
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % dot_interval == 0:
            print("■", end="", flush=True)

    print("] Done")
    return total_loss / total_batches


def evaluate_translation(model, sentence, src_vocab, tgt_vocab):
    """
    Translates a single Romanized Nepali sentence into Devanagari using
    autoregressive greedy decoding (selecting the highest-probability token at each step).
    """
    # Switch to evaluation mode (disables Dropout for deterministic inference)
    model.eval()

    # Encode the input Romanized sentence into token IDs: [<sos>, ..., <eos>]
    encode_fn = src_vocab.encode if hasattr(src_vocab, 'encode') else tgt_vocab.encode
    src_tokens = torch.tensor([encode_fn(sentence)], device=Config.DEVICE)

    # Start target generation with the <sos> token
    tgt_tokens = [Config.SOS_IDX]

    # Limit maximum generated length proportionally to input length to prevent extra word drift
    max_target_tokens = min(Config.MAX_LEN, max(4, int(src_tokens.size(1) * 1.1) + 2))

    # Disable gradient tracking to save RAM and speed up CPU inference
    with torch.no_grad():
        # Encode the source sentence once into contextual memory vectors + padding mask
        memory, src_mask = model.encode(src_tokens)

        # Generate output tokens one by one until <eos> or max_target_tokens is reached
        for _ in range(max_target_tokens):
            tgt_tensor = torch.tensor([tgt_tokens], device=Config.DEVICE)

            # Decode the sequence generated so far against encoder memory
            out = model.decode(tgt_tensor, memory, src_mask)

            # Pick the token ID with the highest logit score at the final sequence position
            next_token = model.fc_out(out)[0, -1].argmax().item()

            # Stop generation if End-Of-Sequence (<eos>) is predicted or length limit is hit
            if next_token == Config.EOS_IDX or len(tgt_tokens) > max_target_tokens:
                break

            tgt_tokens.append(next_token)

    # Convert generated token IDs back into clean Devanagari text
    raw_output = tgt_vocab.decode(tgt_tokens)
    text_clean = re.sub(r'\s+।', '।', raw_output)
    return re.sub(r'\s+', ' ', text_clean).strip()
