import torch
from config import Config

def train_one_epoch(model, dataloader, criterion, optimizer):
    """Executes a training epoch by iterating over mini-batches."""
    model.train()
    total_loss = 0
    
    # Iterate over the smaller chunks provided by the DataLoader
    for src_batch, tgt_batch, src_mask, tgt_mask in dataloader:
        # Move mini-batch tensors to the active hardware (CPU/GPU)
        src_batch = src_batch.to(Config.DEVICE)
        tgt_batch = tgt_batch.to(Config.DEVICE)
        src_mask = src_mask.to(Config.DEVICE)
        tgt_mask = tgt_mask.to(Config.DEVICE)
        
        optimizer.zero_grad()
        
        # Autoregressive shift for teacher forcing
        tgt_input = tgt_batch[:, :-1]
        tgt_expected = tgt_batch[:, 1:]
        current_tgt_mask = tgt_mask[:, :-1]
        
        # Forward pass through the Transformer
        output = model(src_batch, tgt_input, src_mask, current_tgt_mask)
        
        # Flatten outputs to calculate CrossEntropyLoss against the expected tokens
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt_expected.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    # Return the average loss across all mini-batches in this epoch
    return total_loss / len(dataloader)

def evaluate_translation(model, sentence, src_vocab, tgt_vocab):
    """Translates a sentence character-by-character using greedy decoding."""
    model.eval()
    
    # Encode input sentence and add batch dimension [1, seq_len]
    src_tokens = torch.tensor([src_vocab.encode(sentence)]).to(Config.DEVICE)
    src_mask = (src_tokens == Config.PAD_IDX).to(Config.DEVICE)
    
    # Initialize the decoder target with the Start of Sequence token
    tgt_tokens = [Config.SOS_IDX]
    
    with torch.no_grad():
        for _ in range(Config.MAX_LEN):
            tgt_tensor = torch.tensor([tgt_tokens]).to(Config.DEVICE)
            tgt_mask = (tgt_tensor == Config.PAD_IDX).to(Config.DEVICE)
            
            # Predict the next character distributions
            output = model(src_tokens, tgt_tensor, src_mask, tgt_mask)
            
            # Pick the absolute highest probability character index at the last position
            next_token = output[0, -1].argmax().item()
            
            # Stop if the model emits End of Sequence
            if next_token == Config.EOS_IDX:
                break
                
            tgt_tokens.append(next_token)
            
    return tgt_vocab.decode(tgt_tokens)
