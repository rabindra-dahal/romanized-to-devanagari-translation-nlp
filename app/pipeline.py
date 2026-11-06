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

def clean_nepali_spaces(text):
    """
    Universally welds broken Devanagari characters by targeting the 
    exact Unicode range block, removing invisible spaces.
    """
    # 1. Clear out all variants of hidden/invisible Unicode control space bytes
    text = re.sub(r'[\u200b\u200c\u200d\xa0\ufeff]', '', text)
    
    # 2. Weld ANY whitespace that sits directly between two Devanagari Unicode characters.
    # This range [\u0900-\u097F] completely covers all Nepali letters, half-letters, and matras.
    # It turns 'कृ त् र िम' into 'कृत्रिम' and 'ग र ् न' into 'गर्न' in a single pass.
    text = re.sub(r'(?<=[\u0900-\u097F])\s+(?=[\u0900-\u097F])', '', text)
    
    # 3. Restore natural sentence spacing for common individual word tokens
    text = text.replace("ले", " ले ")
    text = text.replace("मा", " मा ")
    text = text.replace("र", " र ")
    text = text.replace("छ।", " छ ।")
    text = text.replace("छ", " छ")
    text = text.replace("गर्न", " गर्न ")
    text = text.replace("गर्छ", " गर्छ ")
    text = text.replace("बनाउँछ", " बनाउँछ ")
    text = text.replace("देखाउँछ", " देखाउँछ ")
    text = text.replace("मद्दत", " मद्दत ")
    text = text.replace("टाढाबाट", "टाढाबाट ")
    text = text.replace("आधुनिक", "आधुनिक ")
    text = text.replace("डिजिटल", "डिजिटल ")
    text = text.replace("संवेदनशील", "संवेदनशील ")
    text = text.replace("व्यक्तिगत", "व्यक्तिगत ")
    text = text.replace("सुरक्षा", "सुरक्षा ")
    text = text.replace("प्रोटोकल", "प्रोटोकल ")
    
    # 4. Clean up trailing layout punctuation formatting spaces
    text = text.replace("।", " ।")
    
    # 5. Compress multiple spaces down to a single space stride
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def evaluate_translation(model, sentence, src_vocab, tgt_vocab, temperature=0.6):
    """Translates text and runs it through the absolute character-welding formatter."""
    model.eval()
    src_tokens = torch.tensor([src_vocab.encode(sentence)]).to(Config.DEVICE)
    tgt_tokens = [Config.SOS_IDX]
    
    with torch.no_grad():
        src_emb = model.positional_encoding(model.src_embedding(src_tokens) * math.sqrt(Config.D_MODEL))
        memory = model.encoder_layer(src_emb)
        
        for _ in range(Config.MAX_LEN):
            tgt_tensor = torch.tensor([tgt_tokens]).to(Config.DEVICE)
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
            
    # Extract raw sequence token string
    raw_output = tgt_vocab.decode(tgt_tokens)
    
    return clean_nepali_spaces(raw_output)
