"""
Model manager for loading and managing the LLM
"""
import time
import torch
from typing import List, Dict, Any, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig
)
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages the LLM model loading, caching, and inference"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = settings.model_name
        self.device = torch.device(settings.device)
        self.is_loaded = False
        self.load_time = None
        
        # Set torch threads for CPU optimization
        if settings.device == "cpu":
            torch.set_num_threads(settings.torch_threads)
    
    async def load_model(self) -> bool:
        """Load the model and tokenizer"""
        try:
            start_time = time.time()
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=settings.model_cache_dir,
                padding_side="left"
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=settings.model_cache_dir,
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info(f"Model loaded successfully in {self.load_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.is_loaded = False
            return False
    
    async def generate_text(
        self,
        prompt: str,
        max_length: int = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> Dict[str, Any]:
        """Generate text based on the input prompt"""
        
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")
        
        start_time = time.time()
        
        # Use defaults if not provided
        max_length = max_length or settings.default_max_length
        temperature = temperature or settings.default_temperature
        top_p = top_p or settings.default_top_p
        top_k = top_k or settings.default_top_k
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=settings.max_model_length
            ).to(self.device)
            
            # Configure generation parameters
            generation_config = GenerationConfig(
                max_length=min(inputs.input_ids.shape[1] + max_length, settings.max_model_length),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    generation_config=generation_config
                )
            
            # Decode generated text
            generated_texts = []
            for output in outputs:
                # Remove the input tokens from the output
                generated_tokens = output[inputs.input_ids.shape[1]:]
                generated_text = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                )
                generated_texts.append(generated_text)
            
            generation_time = time.time() - start_time
            
            return {
                "generated_text": generated_texts,
                "prompt": prompt,
                "model_name": self.model_name,
                "generation_time": generation_time,
                "parameters": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "num_return_sequences": num_return_sequences,
                    "do_sample": do_sample
                }
            }
            
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise RuntimeError(f"Text generation failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        
        if not self.is_loaded:
            return {
                "model_name": self.model_name,
                "model_type": "Unknown",
                "model_size": "Unknown",
                "device": str(self.device),
                "max_length": settings.max_model_length,
                "loaded": False
            }
        
        # Calculate approximate model size
        total_params = sum(p.numel() for p in self.model.parameters())
        model_size_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32
        
        return {
            "model_name": self.model_name,
            "model_type": self.model.config.model_type if hasattr(self.model, 'config') else "Unknown",
            "model_size": f"~{model_size_mb:.1f}MB ({total_params:,} parameters)",
            "device": str(self.device),
            "max_length": settings.max_model_length,
            "loaded": True
        }


# Global model manager instance
model_manager = ModelManager()