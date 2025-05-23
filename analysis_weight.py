import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "3"
import torch
from pathlib import Path
import json
import numpy as np
from safetensors import safe_open
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('llama_analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_model_weights(model_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Load tokenizer and model
    model_name = "yahma/llama-7b-hf" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

def analyze_weight_matrix(weight_tensor, name):
    """Analyze statistics of a weight matrix"""
    stats = {
        "name": name,
        "shape": list(weight_tensor.shape),
        "mean": float(torch.mean(weight_tensor).item()),
        "std": float(torch.std(weight_tensor).item()),
        "min": float(torch.min(weight_tensor).item()),
        "max": float(torch.max(weight_tensor).item()),
        "sparsity": float((weight_tensor == 0).sum().item() / weight_tensor.numel()),
        "total_params": weight_tensor.numel()
    }
    return stats

def save_statistics(stats, output_file):
    """Save statistics to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistics saved to: {output_file}")

def main(model_path, output_file="llama_weight_stats.json"):
    try:
        # Load model weights
        weights = load_model_weights(model_path)
        
        # Analyze each weight matrix
        stats = []
        total_params = 0
        
        for name, tensor in weights.items():
            if isinstance(tensor, torch.Tensor) and len(tensor.shape) >= 2:
                logger.info(f"Analyzing layer: {name}")
                layer_stats = analyze_weight_matrix(tensor, name)
                stats.append(layer_stats)
                total_params += layer_stats["total_params"]
        
        # Add summary statistics
        summary = {
            "model_path": str(model_path),
            "total_parameters": total_params,
            "total_layers_analyzed": len(stats),
            "layer_statistics": stats
        }
        
        # Save results
        save_statistics(summary, output_file)
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze LLaMA model weights")
    parser.add_argument("--model_path", default=".", type=str, help="Path to the model file (.bin or .safetensors)")
    parser.add_argument("--output", type=str, default="llama_weight_stats.json",
                        help="Output JSON file path")
    
    args = parser.parse_args()
    logger = setup_logging()
    main(Path(args.model_path), args.output)