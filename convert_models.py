import os
import subprocess
import shutil
import mlx.core as mx
from huggingface_hub import snapshot_download

def convert_embedding_model(model_id, output_path):
    print(f"Converting embedding model {model_id} to {output_path}...")
    # Use mlx-embeddings conversion tool
    cmd = [
        "python", "-m", "mlx_embeddings.convert",
        "--hf-path", model_id,
        "--mlx-path", output_path,
        "--dtype", "float16"
    ]
    subprocess.run(cmd, check=True)
    
    # Extract sparse linear weights for BGE-M3
    print("Extracting sparse weights for BGE-M3...")
    import torch
    
    # Load original model to get the linear layer
    temp_dir = "temp_hf_model"
    snapshot_download(repo_id=model_id, local_dir=temp_dir)
    
    # BGE-M3 has a specific architecture in FlagEmbedding, but it's basically XLM-R 
    # with extra heads. We just need the weights for the sparse head.
    # In some versions it's called 'colbert_linear' or similar.
    # Let's try to find it in the state dict.
    state_dict = torch.load(os.path.join(temp_dir, "pytorch_model.bin"), map_location="cpu")
    
    sparse_weight = None
    for key in state_dict:
        if "sparse_linear.weight" in key or "colbert_linear.weight" in key:
            sparse_weight = state_dict[key]
            break
            
    if sparse_weight is not None:
        # Convert torch to mlx
        sparse_weight_mlx = mx.array(sparse_weight.numpy())
        mx.savez(os.path.join(output_path, "sparse_linear.safetensors"), weight=sparse_weight_mlx)
        print(f"Saved sparse weights to {output_path}/sparse_linear.safetensors")
    else:
        print("Could not find sparse weights in state dict. Trying alternative names...")
        # Fallback: check if it's in another file or named differently
    
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    # BGE-M3 Embedding
    convert_embedding_model("BAAI/bge-m3", "models/bge-m3-mlx")
    
    print("Model conversion complete!")
