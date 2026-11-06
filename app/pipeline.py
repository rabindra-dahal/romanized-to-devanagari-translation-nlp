import torch
from config import Config

def train_one_epoch(model, dataloader, criterion, optimizer):
    """Executes a training epoch by iterating over mini-batches."""
    model.train()
    total_loss = 0
    
    for src_batch, tgt_batch, src_mask, tgt_mask in dataloader:
        src_batch = src_batch.to(Config.DEVICE)
        tgt_batch = tgt_batch.to(Config.DEVICE)
        src_mask = src_mask.to(Config.DEVICE)
        tgt_mask = tgt_mask.to(Config.DEVICE)
        
        optimizer.zero_grad()
        
        tgt_input = tgt_batch[:, :-1]
        tgt_expected = tgt_batch[:, 1:]
        current_tgt_mask = tgt_mask[:, :-1]
        
        output = model(src_batch, tgt_input, src_mask, current_tgt_mask)
        
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt_expected.reshape(-1))
        loss.backward()
        
        # Protect weights against spiking gradient updates
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate_translation(model, sentence, src_vocab, tgt_vocab):
    """Translates a sentence character-by-character using clean evaluation tracking."""
    model.eval()
    
    src_tokens = torch.tensor([src_vocab.encode(sentence)]).to(Config.DEVICE)
    src_mask = (src_tokens == Config.PAD_IDX).to(Config.DEVICE)
    
    tgt_tokens = [Config.SOS_IDX]
    
    with torch.no_grad():
        for _ in range(Config.MAX_LEN):
            tgt_tensor = torch.tensor([tgt_tokens]).to(Config.DEVICE)
            # Crucial: During step-by-step inference generation, we are not passing 
            # padding blocks to our decoder, so our key tracking mask should be completely False.
            tgt_mask = torch.zeros_like(tgt_tensor, dtype=torch.bool).to(Config.DEVICE)
            
            output = model(src_tokens, tgt_tensor, src_mask, tgt_mask)
            
            # Isolate the exact trailing token projection probabilities
            next_token = output[0, -1].argmax().item()
            
            if next_token == Config.EOS_IDX:
                break
                
            tgt_tokens.append(next_token)
            
            # Safety brake: Stop generation if it gets trapped in an infinite loop
            if len(tgt_tokens) > len(sentence) * 3 + 5:
                break
            
    return tgt_vocab.decode(tgt_tokens)
