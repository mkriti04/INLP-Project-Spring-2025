import torch
import torch.nn.functional as F
from model_with_negative_handled import Transformer
from utils import load_checkpoint, calculate_accuracy


class ArithmeticTransformer:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.vocab = checkpoint['vocab']
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}
        
        # Initialize model with the same parameters as during training
        config = checkpoint['config']
        self.model = Transformer(
            vocab_size=len(self.vocab),
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=0.0  # Use 0 for inference
        ).to(device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
    
    def tokenize(self, problem):
        """Convert problem string to tensor of indices."""
        tokens = ['<sos>'] + list(problem) + ['<eos>']
        indices = [self.vocab.get(token, self.vocab['<pad>']) for token in tokens]
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def detokenize(self, indices):
        """Convert tensor of indices to string."""
        # Filter out special tokens and convert to string
        tokens = [self.idx_to_token.get(idx.item(), '') for idx in indices]
        return ''.join([t for t in tokens if t not in ['<sos>', '<eos>', '<pad>']])
    
    def predict(self, problem, max_len=100):  # Increase max_len for larger numbers
        """Generate solution for given problem."""
        self.model.eval()  # Ensure model is in evaluation mode
        
        # Tokenize the problem
        src = self.tokenize(problem)
        
        # Initialize target with <sos> token
        tgt = torch.tensor([[self.vocab['<sos>']]], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_len):
                # Create masks
                tgt_mask = self.model.generate_square_subsequent_mask(tgt.size(1)).to(self.device)
                
                # Forward pass
                output = self.model(
                    src=src,
                    tgt=tgt,
                    src_mask=None,
                    tgt_mask=tgt_mask,
                    memory_mask=None,
                    src_key_padding_mask=None,
                    tgt_key_padding_mask=None,
                    memory_key_padding_mask=None
                )
                
                # Get the next token prediction
                next_token_logits = output[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
                # Append to target sequence
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Stop if <eos> token is generated
                if next_token.item() == self.vocab['<eos>']:
                    break
        
        # Get the generated sequence without <sos> and <eos>
        result = self.detokenize(tgt[0][1:])
        
        return result



def calculate_char_accuracy(pred, solution):
    """Calculate character-level accuracy between prediction and solution."""
    if len(pred) == 0 or len(solution) == 0:
        return 0.0
    
    # Truncate to the shorter length for comparison
    min_len = min(len(pred), len(solution))
    matches = sum(p == s for p, s in zip(pred[:min_len], solution[:min_len]))
    
    # Penalize for length differences
    total_chars = max(len(pred), len(solution))
    
    return matches / total_chars

def calculate_perplexity(model, problem, solution, vocab):
    """Calculate perplexity of the model on a given problem-solution pair."""
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize input and target
    src_tokens = ['<sos>'] + list(problem) + ['<eos>']
    src_indices = [vocab.get(token, vocab['<pad>']) for token in src_tokens]
    src = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)
    
    # For perplexity calculation, we need to handle negative solutions
    # by removing the negative sign since the model wasn't trained on it
    solution_for_model = solution
    if solution.startswith('-'):
        solution_for_model = solution[1:]
    
    tgt_tokens = ['<sos>'] + list(solution_for_model) + ['<eos>']
    tgt_indices = [vocab.get(token, vocab['<pad>']) for token in tgt_tokens]
    tgt = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(device)
    
    # Create shifted target for input to decoder (remove last token)
    tgt_input = tgt[:, :-1]
    
    # Target for loss calculation (remove first token)
    tgt_output = tgt[:, 1:]
    
    with torch.no_grad():
        # Create masks
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        
        # Forward pass
        output = model(
            src=src,
            tgt=tgt_input,
            src_mask=None,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None
        )
        
        # Calculate cross entropy loss
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)
        
        loss = F.cross_entropy(output, tgt_output, reduction='mean')
        
        # Perplexity = exp(loss)
        perplexity = torch.exp(loss).item()
        
    return perplexity

def test_model(model_path, test_problems, test_solutions):
    """Test model on test set with multiple metrics."""
    model = ArithmeticTransformer(model_path)
    exact_matches = 0
    total_char_accuracy = 0.0
    total_perplexity = 0.0
    total = len(test_problems)
    
    for i, (problem, solution) in enumerate(zip(test_problems, test_solutions)):
        # Correct the expected solution if it's a subtraction with first number smaller
        corrected_solution = solution
        if '-' in problem:
            parts = problem.split('-')
            if len(parts) == 2:
                try:
                    num1 = int(parts[0])
                    num2 = int(parts[1])
                    if num1 < num2:
                        # The expected result should be negative
                        corrected_solution = '-' + solution
                        # Update the test_solutions list for future reference
                        test_solutions[i] = corrected_solution
                except ValueError:
                    pass  # Not valid numbers, continue with normal comparison
        
        # Get model prediction
        pred = model.predict(problem)
        print(f"Problem: {problem} | Expected: {corrected_solution} | Got: {pred}")
        # Calculate exact match with corrected solution
        if pred == corrected_solution:
            exact_matches += 1
        
        # Calculate character-level accuracy with corrected solution
        char_acc = calculate_char_accuracy(pred, corrected_solution)
        total_char_accuracy += char_acc
        
        # Calculate perplexity with corrected solution
        perplexity = calculate_perplexity(model.model, problem, corrected_solution, model.vocab)
        total_perplexity += perplexity
    
    # Calculate average metrics
    exact_match_accuracy = exact_matches / total
    avg_char_accuracy = total_char_accuracy / total
    avg_perplexity = total_perplexity / total
    
    print(f"Exact Match Accuracy: {exact_match_accuracy:.2%}")
    print(f"Character-level Accuracy: {avg_char_accuracy:.2%}")
    print(f"Average Perplexity: {avg_perplexity:.4f} (lower is better)")
    
    return {
        'exact_match': exact_match_accuracy,
        'char_accuracy': avg_char_accuracy,
        'perplexity': avg_perplexity
    }

if __name__ == '__main__':
    # Example usage
    model_path = 'models/final_model.pth'
    
    print("Within range")
    
    # Generate some test data
    from data_generator import generate_dataset
    test_problems, test_solutions = generate_dataset(100, max_digits=3)
    
    # Test the model with multiple metrics
    metrics = test_model(model_path, test_problems[:20], test_solutions[:20])
    
    print("\nSummary of Metrics:")
    print(f"Exact Match Accuracy: {metrics['exact_match']:.2%}")
    print(f"Character-level Accuracy: {metrics['char_accuracy']:.2%}")
    print(f"Perplexity: {metrics['perplexity']:.4f}")
    
    print("===============================================================")
    print("\Out of range")
    
    from data_generator import generate_dataset
    test_problems, test_solutions = generate_dataset(100, max_digits=4)
    
    # Test the model with multiple metrics
    metrics = test_model(model_path, test_problems[:20], test_solutions[:20])
    
    print("\nSummary of Metrics:")
    print(f"Exact Match Accuracy: {metrics['exact_match']:.2%}")
    print(f"Character-level Accuracy: {metrics['char_accuracy']:.2%}")
    print(f"Perplexity: {metrics['perplexity']:.4f}")
