import torch
from config import Config

def train_one_epoch(model, src_batch, tgt_batch, src_mask, tgt_mask, criterion, optimizer):
    """Executes a single step of supervised backpropagation training."""
    model.train()         # Toggle internal layers (like Dropout) into active training state
    optimizer.zero_grad() # Clear out accumulated parameter gradients from the previous step
    
    # Target shift logic for teacher forcing:
    tgt_input = tgt_batch[:, :-1]          # Inputs omit the trailing `<eos>` token
    tgt_expected = tgt_batch[:, 1:]        # Output predictions are scored against tokens shifted forward by one
    current_tgt_mask = tgt_mask[:, :-1]    # Adjust the padding mask to align with the shorter inputs
    
    # Process inputs through the model network architecture
    output = model(src_batch, tgt_input, src_mask, current_tgt_mask)
    
    # Reshape the output matrix into 2D to calculate cross-entropy against target token strings
    loss = criterion(output.reshape(-1, output.shape[-1]), tgt_expected.reshape(-1))
    loss.backward()        # Calculate gradients across all network parameters via backpropagation
    optimizer.step()       # Apply calculated gradients to update weights via the Adam optimizer
    return loss.item()     # Return loss as a standard Python floating-point number

def evaluate_translation(model, sentence, src_vocab, tgt_vocab):
    """Translates a single Romanized string using standard greedy search decoding."""
    model.eval()           # Deactivate dropout and non-deterministic behavior for testing
    # Convert text sequence to a batch matrix containing a single row
    src_tokens = torch.tensor([src_vocab.encode(sentence)]).to(Config.DEVICE)
    src_mask = (src_tokens == Config.PAD_IDX).to(Config.DEVICE)
    
    # Prime the target output tracking array with the start-of-sequence token index
    tgt_tokens = [Config.SOS_IDX]
    
    with torch.no_grad():  # Turn off the tracking engine to save runtime memory
        for _ in range(Config.MAX_LEN):
            # Wrap standard token sequences inside a Torch device tensor matrix
            tgt_tensor = torch.tensor([tgt_tokens]).to(Config.DEVICE)
            tgt_mask = (tgt_tensor == Config.PAD_IDX).to(Config.DEVICE)
            
            # Predict subsequent probability spaces based on active contexts
            output = model(src_tokens, tgt_tensor, src_mask, tgt_mask)
            # Find the vocabulary token index with the highest probability value at the final sequence step
            next_token = output[0, -1].argmax().item()
            
            # Immediately exit loop if model emits the end-of-sequence marker
            if next_token == Config.EOS_IDX:
                break
            # Append the predicted character token index to the output tracking sequence
            tgt_tokens.append(next_token)
            
    # Unpack token integers back into a readable Devanagari string
    return tgt_vocab.decode(tgt_tokens)
