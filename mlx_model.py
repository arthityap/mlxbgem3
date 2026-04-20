import mlx.core as mx
import mlx.nn as nn
import numpy as np
import json
import os
from transformers import AutoTokenizer
from mlx_embeddings.utils import load as load_mlx_model
from typing import List, Dict, Tuple, Any

class MLXBGEM3Model:
    def __init__(self, model_path: str):
        self.model, self.tokenizer = load_mlx_model(model_path)
        self.sparse_linear = None
        try:
            weights = mx.load(f"{model_path}/sparse_linear.safetensors")
            self.sparse_linear = nn.Linear(1024, 1, bias=False)
            self.sparse_linear.update({"weight": weights["weight"]})
        except:
            print("Sparse weights not found, hybrid embedding will be unavailable")

    def encode_dense(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="mlx", max_length=8192)
            outputs = self.model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
            all_embeddings.append(np.array(outputs.text_embeds))
        return np.concatenate(all_embeddings, axis=0)

    def encode_sparse(self, texts: List[str], batch_size: int = 4) -> List[Dict[str, float]]:
        if self.sparse_linear is None:
            return [{} for _ in texts]
        
        all_sparse = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="mlx", max_length=8192)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            
            lexical_weights = mx.maximum(self.sparse_linear(last_hidden_state), 0)
            lexical_weights = lexical_weights.squeeze(-1) * attention_mask
            
            for b in range(lexical_weights.shape[0]):
                weights_dict = {}
                for t in range(lexical_weights.shape[1]):
                    w = float(lexical_weights[b, t])
                    if w > 0:
                        token_id = int(input_ids[b, t])
                        token = self.tokenizer.convert_ids_to_tokens(token_id)
                        if token not in weights_dict or w > weights_dict[token]:
                            weights_dict[token] = w
                all_sparse.append(weights_dict)
        return all_sparse

class XLMRobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.out_proj = nn.Linear(config["hidden_size"], config.get("num_labels", 1))

    def __call__(self, x):
        x = self.dense(x)
        x = mx.tanh(x)
        x = self.out_proj(x)
        return x

class MLXReranker:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        with open(os.path.join(model_path, "config.json"), "r") as f:
            self.config = json.load(f)
        
        from mlx_embeddings.models.xlm_roberta import Model as XLMRobertaBaseModel
        from mlx_embeddings.models.xlm_roberta import ModelArgs
        
        import inspect
        sig = inspect.signature(ModelArgs)
        model_args_dict = {k: v for k, v in self.config.items() if k in sig.parameters}
        model_args = ModelArgs(**model_args_dict)
        self.roberta = XLMRobertaBaseModel(model_args)
        self.classifier = XLMRobertaClassificationHead(self.config)
        
        weights = mx.load(os.path.join(model_path, "model.safetensors"))
        
        roberta_weights = {}
        classifier_weights = {}
        for k, v in weights.items():
            if k.startswith("roberta."):
                roberta_weights[k[8:]] = v
            elif k.startswith("classifier."):
                classifier_weights[k[11:]] = v
            else:
                roberta_weights[k] = v
        
        # Fill missing pooler weights if necessary
        if "pooler.dense.weight" not in roberta_weights:
            roberta_weights["pooler.dense.weight"] = mx.zeros((self.config["hidden_size"], self.config["hidden_size"]))
            roberta_weights["pooler.dense.bias"] = mx.zeros((self.config["hidden_size"],))

        self.roberta.load_weights(list(roberta_weights.items()))
        self.classifier.load_weights(list(classifier_weights.items()))

    def compute_score(self, pairs: List[List[str]], batch_size: int = 4) -> List[float]:
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="mlx", max_length=512)
            
            outputs = self.roberta(inputs["input_ids"], attention_mask=inputs["attention_mask"])
            hidden = outputs.last_hidden_state
            cls_token_state = hidden[:, 0, :]
            logits = self.classifier(cls_token_state)
            
            if logits.shape[-1] == 1:
                scores = mx.sigmoid(logits).squeeze(-1)
            else:
                scores = mx.softmax(logits, axis=-1)[:, 1]
            
            all_scores.extend(scores.tolist())
        return all_scores
