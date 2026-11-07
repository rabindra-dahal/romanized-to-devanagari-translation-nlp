import torch
import math
import re
from config import Config

def train_one_epoch(model, dataloader, criterion, optimizer, epoch):
    """Runs a single training iteration step with an inline epoch indicator map."""
    model.train()
    total_loss = 0
    total_batches = len(dataloader)
    
    print(f"  Epoch {epoch:02d}/40 Progress: [", end="", flush=True)
    dot_interval = max(1, total_batches // 12)
    
    for batch_idx, (src_batch, tgt_batch) in enumerate(dataloader):
        src_batch = src_batch.to(Config.DEVICE)
        tgt_batch = tgt_batch.to(Config.DEVICE)
        
        optimizer.zero_grad()
        tgt_input = tgt_batch[:, :-1]
        tgt_expected = tgt_batch[:, 1:]
        
        output = model(src_batch, tgt_input)
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt_expected.reshape(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % dot_interval == 0:
            print("■", end="", flush=True)
            
    print("] Done")
    return total_loss / total_batches


def evaluate_translation(model, sentence, src_vocab, tgt_vocab, temperature=0.2):
    """Translates text using low-temperature sampling and delegates welding to SubwordVocabulary."""
    model.eval()
    
    encode_fn = src_vocab.encode if hasattr(src_vocab, 'encode') else tgt_vocab.encode
    src_tokens = torch.tensor([encode_fn(sentence)], device=Config.DEVICE)
    tgt_tokens = [Config.SOS_IDX]
    
    with torch.no_grad():
        src_emb = model.positional_encoding(model.src_embedding(src_tokens) * math.sqrt(Config.D_MODEL))
        memory = model.encoder_layer(src_emb)
        
        for _ in range(Config.MAX_LEN):
            tgt_tensor = torch.tensor([tgt_tokens], device=Config.DEVICE)
            tgt_emb = model.positional_encoding(model.tgt_embedding(tgt_tensor) * math.sqrt(Config.D_MODEL))
            
            mask = torch.tril(torch.ones(tgt_tensor.size(1), tgt_tensor.size(1), device=Config.DEVICE))
            out = model.decoder_layer(tgt_emb, memory, tgt_mask=mask)
            
            logits = model.fc_out(out)[0, -1]
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == Config.EOS_IDX or len(tgt_tokens) > Config.MAX_LEN:
                break
            tgt_tokens.append(next_token)
            
    # Delegate decoding and orthographic welding to the SubwordVocabulary class pipeline
    raw_output = tgt_vocab.decode(tgt_tokens)
    
    # Keep Purna Viram attached cleanly to the last word without a leading space
    text_clean = re.sub(r'\s+।', '।', raw_output)
    clean_sentence = re.sub(r'\s+', ' ', text_clean).strip()
    
    return clean_sentence
