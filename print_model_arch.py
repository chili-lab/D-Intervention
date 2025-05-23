import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def print_model_architecture(model_name):
    """
    Load a model from Hugging Face and print its architecture.
    
    Args:
        model_name (str): The name of the model on Hugging Face.
    """
    print(f"Loading model: {model_name}")
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 to reduce memory usage
        device_map="auto",           # Automatically place model on available devices
        trust_remote_code=True       # Required for some models
    )
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    
    # # Print parameter count
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # print(f"\nTotal parameters: {total_params:,}")
    # print(f"Trainable parameters: {trainable_params:,}")
    
    # # Print model config
    # print("\nModel Config:")
    # print(model.config)
    
    # print model architecture
    print(model.config.architectures[0].lower())
    
    # Print tokenizer information
    print("\nTokenizer Information:")
    print(f"BOS token ID: {tokenizer.bos_token_id}")
    print(f"BOS token: '{tokenizer.bos_token}'")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"EOS token: '{tokenizer.eos_token}'")
    
    return model, tokenizer

if __name__ == "__main__":
    # Load and print architecture for Qwen/Qwen2.5-3B
    model_name = "google/gemma-3-12b-it"
    model, tokenizer = print_model_architecture(model_name)
