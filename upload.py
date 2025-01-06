from huggingface_hub import HfApi
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

def push_to_hub(model_path, tokenizer_path, repo_name):
    # Initialize the API
    api = HfApi()
    
    # First load the base BART model
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    
    # Load your fine-tuned state dict - NOW WITH CPU MAPPING
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    # Push to hub
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)

if __name__ == "__main__":
    model_path = "models/bart_state.pt"
    tokenizer_path = "models/tokenizer"
    repo_name = "Lord-Connoisseur/headline-generator"
    
    push_to_hub(model_path, tokenizer_path, repo_name)