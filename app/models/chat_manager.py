"""
Chat-aware model manager that handles conversation context and chat templates
"""
import time
import torch
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,
    TextIteratorStreamer
)
import logging
from jinja2 import Template
import asyncio
from threading import Thread

from app.core.config import settings, get_model_profile, CHAT_TEMPLATES
from app.models.conversation import Conversation

logger = logging.getLogger(__name__)


class ModelLoadingProgress:
    """Class to track model loading progress"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.stage = "initializing"
        self.progress = 0.0
        self.status_message = "Starting model load..."
        self.start_time = time.time()
        self.callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable):
        """Add a progress callback function"""
        self.callbacks.append(callback)
    
    def update(self, stage: str, progress: float, message: str = ""):
        """Update progress and notify callbacks"""
        self.stage = stage
        self.progress = max(0.0, min(100.0, progress))
        self.status_message = message or f"{stage.title()}..."
        
        # Notify all callbacks
        for callback in self.callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since start"""
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "model_name": self.model_name,
            "stage": self.stage,
            "progress": self.progress,
            "status_message": self.status_message,
            "elapsed_time": self.get_elapsed_time()
        }


class ChatModelManager:
    """Enhanced model manager with chat support and conversation context"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.model_name
        self.profile = get_model_profile(self.model_name)
        
        self.model = None
        self.tokenizer = None
        self.device = torch.device(settings.device)
        self.is_loaded = False
        self.load_time = None
        
        # Chat-specific attributes
        self.supports_chat = self.profile.supports_chat
        self.chat_template = None
        
        # Set torch threads for CPU optimization
        if settings.device == "cpu":
            torch.set_num_threads(settings.torch_threads)
    
    async def load_model(self, progress_callback: Optional[Callable] = None) -> bool:
        """Load the model and tokenizer with chat template support and progress tracking"""
        progress = ModelLoadingProgress(self.profile.name)
        if progress_callback:
            progress.add_callback(progress_callback)
        
        try:
            start_time = time.time()
            logger.info(f"Loading model: {self.profile.name} ({self.profile.model_id})")
            
            # Stage 1: Initialize tokenizer
            progress.update("tokenizer", 10, f"Loading tokenizer for {self.profile.name}")
            await asyncio.sleep(0.1)  # Allow UI updates
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.profile.model_id,
                cache_dir=settings.model_cache_dir,
                padding_side="left",
                trust_remote_code=True
            )
            
            progress.update("tokenizer", 25, "Tokenizer loaded, configuring...")
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Setup chat template
            progress.update("template", 30, "Setting up chat template...")
            self._setup_chat_template()
            
            # Stage 2: Prepare model loading
            progress.update("model_prep", 35, "Preparing model configuration...")
            await asyncio.sleep(0.1)
            
            # Load model with appropriate settings for the device
            model_kwargs = {
                "cache_dir": settings.model_cache_dir,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }
            
            # Configure dtype and device-specific settings
            if settings.device == "cuda" and torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
            elif settings.device == "mps" and torch.backends.mps.is_available():
                # Use MPS (Metal Performance Shaders) on Mac
                model_kwargs["torch_dtype"] = torch.float16
            else:
                # CPU fallback - use float32 for better compatibility
                model_kwargs["torch_dtype"] = torch.float32
                
            # Special handling for DeepSeek models on CPU/Mac
            if "deepseek" in self.profile.model_id.lower():
                model_kwargs["torch_dtype"] = torch.float32  # Force float32 for DeepSeek on CPU
                if settings.device != "cuda":
                    # Disable any quantization for CPU
                    model_kwargs["load_in_8bit"] = False
                    model_kwargs["load_in_4bit"] = False
            
            # Stage 3: Download/load model (this is the longest step)
            progress.update("downloading", 40, f"Downloading {self.profile.name} model files...")
            
            # For large models, provide more detailed feedback
            estimated_size_gb = self.profile.memory_gb
            if estimated_size_gb > 10:
                progress.update("downloading", 50, f"Large model detected ({estimated_size_gb}GB), this may take several minutes...")
                await asyncio.sleep(0.5)
                progress.update("downloading", 60, "Please wait while model files are downloaded and cached...")
                await asyncio.sleep(0.5)
                progress.update("downloading", 70, "Download in progress... (first time only)")
                await asyncio.sleep(0.5)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.profile.model_id,
                **model_kwargs
            )
            
            progress.update("loading", 80, "Model downloaded, loading into memory...")
            await asyncio.sleep(0.1)
            
            # Stage 4: Move to device and finalize
            progress.update("device", 85, f"Moving model to {self.device}...")
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            progress.update("finalizing", 95, "Finalizing model setup...")
            await asyncio.sleep(0.1)
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            
            progress.update("complete", 100, f"Model loaded successfully in {self.load_time:.1f}s")
            
            logger.info(f"Model loaded successfully in {self.load_time:.2f} seconds")
            logger.info(f"Chat support: {self.supports_chat}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Memory requirement: {self.profile.memory_gb}GB")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            progress.update("error", 0, error_msg)
            logger.error(error_msg)
            self.is_loaded = False
            return False
    
    def _setup_chat_template(self) -> None:
        """Setup chat template for the model"""
        if not self.profile.supports_chat:
            return
        
        # For specific models, always use our custom templates
        force_custom_template = [
            "microsoft/DialoGPT-medium",
            "gpt2", 
            "gpt2-large",
            "gpt2-medium",
            "distilgpt2",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct", 
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "deepseek-ai/deepseek-coder-1.3b-instruct",
            "deepseek-ai/deepseek-coder-6.7b-instruct",
            "deepseek-ai/deepseek-math-7b-instruct"
        ]
        
        # Use custom templates for models that need them
        if self.profile.model_id in force_custom_template or not hasattr(self.tokenizer, 'chat_template') or not self.tokenizer.chat_template:
            template_name = self.profile.chat_template or "default"
            if template_name in CHAT_TEMPLATES:
                self.tokenizer.chat_template = CHAT_TEMPLATES[template_name]
                logger.info(f"Using custom chat template: {template_name}")
            else:
                logger.warning(f"Chat template '{template_name}' not found, using default")
                self.tokenizer.chat_template = CHAT_TEMPLATES["default"]
        else:
            # Use model's built-in chat template
            logger.info("Using model's built-in chat template")
    
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
        """Generate text for regular (non-chat) use"""
        
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")
        
        start_time = time.time()
        
        # Use profile defaults if not provided
        max_length = max_length or self.profile.default_max_tokens
        temperature = temperature or self.profile.default_temperature
        top_p = top_p or self.profile.default_top_p
        top_k = top_k or self.profile.default_top_k
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.profile.max_length
            ).to(self.device)
            
            # Configure generation parameters
            generation_config = GenerationConfig(
                max_new_tokens=max_length,
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
                "model_name": self.profile.name,
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
    
    async def chat(
        self,
        conversation: Conversation,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate chat response for a conversation"""
        
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")
        
        if not self.supports_chat:
            # Fall back to regular text generation for non-chat models
            last_message = conversation.get_last_user_message()
            if not last_message:
                raise ValueError("No user message found in conversation")
            
            result = await self.generate_text(
                prompt=last_message,
                max_length=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
            
            # Add response to conversation
            response_text = result["generated_text"][0]
            conversation.add_assistant_message(response_text)
            
            # Return the result directly
            return {
                "response": response_text,
                "conversation": conversation,
                "generation_time": result["generation_time"],
                "model_name": self.profile.name,
                "parameters": result["parameters"],
                "finished": True
            }
        
        start_time = time.time()
        
        # Use profile defaults if not provided
        max_tokens = max_tokens or self.profile.default_max_tokens
        temperature = temperature or self.profile.default_temperature
        top_p = top_p or self.profile.default_top_p
        top_k = top_k or self.profile.default_top_k
        
        try:
            # Format conversation using chat template
            messages = conversation.get_messages_for_model()
            
            if hasattr(self.tokenizer, 'apply_chat_template'):
                # Use tokenizer's chat template
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fall back to manual template application
                template = Template(self.tokenizer.chat_template)
                prompt = template.render(messages=messages, add_generation_prompt=True)
            
            # Tokenize the formatted prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.profile.max_length - max_tokens  # Leave room for response
            ).to(self.device)
            
            # Configure generation
            generation_config = GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                repetition_penalty=1.1,  # Reduce repetition
                no_repeat_ngram_size=3,  # Avoid repeating 3-grams
                early_stopping=True      # Stop at natural ending points
            )
            
            # Non-streaming generation for now (to avoid async generator issues)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    generation_config=generation_config
                )
            
            # Decode response
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            response_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()
            
            # Post-process response to clean up common issues
            response_text = self._clean_response(response_text, conversation)
            
            # Ensure we have a meaningful response
            if not response_text or len(response_text.strip()) < 3:
                response_text = "I understand. Could you please rephrase your question?"
            
            # Add to conversation
            conversation.add_assistant_message(response_text)
            generation_time = time.time() - start_time
            
            # Return the result directly
            return {
                "response": response_text,
                "conversation": conversation,
                "generation_time": generation_time,
                "model_name": self.profile.name,
                "parameters": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k
                },
                "finished": True
            }
                
        except Exception as e:
            logger.error(f"Chat generation failed: {str(e)}")
            raise RuntimeError(f"Chat generation failed: {str(e)}")
    
    def _clean_response(self, response_text: str, conversation: Conversation) -> str:
        """Clean up the generated response to remove common issues"""
        if not response_text:
            return ""
        
        # Remove common prefixes that models sometimes add
        prefixes_to_remove = [
            "Bot:", "AI:", "Assistant:", "AI Assistant:", 
            "Human:", "User:", "You:", "<|endoftext|>",
            "The following is", "Here is", "This is"
        ]
        
        original_response = response_text
        for prefix in prefixes_to_remove:
            if response_text.startswith(prefix):
                response_text = response_text[len(prefix):].strip()
                break
        
        # Remove repetitive patterns (common with some models)
        lines = response_text.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and line not in clean_lines[-3:]:  # Avoid immediate repetition
                clean_lines.append(line)
        
        response_text = '\n'.join(clean_lines).strip()
        
        # Handle cases where model repeats the user's message
        last_user_message = conversation.get_last_user_message()
        if last_user_message and response_text.lower().startswith(last_user_message.lower()[:20]):
            # Find where the user message ends and new content begins
            user_msg_words = last_user_message.split()[:5]  # First 5 words
            response_words = response_text.split()
            
            # Find where the repetition ends
            for i, word in enumerate(response_words):
                if i >= len(user_msg_words) or word.lower() != user_msg_words[i].lower():
                    response_text = ' '.join(response_words[i:]).strip()
                    break
        
        # Ensure response doesn't start with punctuation
        while response_text and response_text[0] in ".,!?;:":
            response_text = response_text[1:].strip()
        
        return response_text or original_response
    
    async def _generate_stream(
        self, 
        inputs: Dict[str, torch.Tensor], 
        generation_config: GenerationConfig
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "generation_config": generation_config,
            "streamer": streamer
        }
        
        # Start generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield chunks as they become available
        for chunk in streamer:
            yield chunk
        
        # Wait for generation to complete
        thread.join()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        
        base_info = {
            "model_name": self.profile.name,
            "model_id": self.profile.model_id,
            "model_type": "Unknown",
            "device": str(self.device),
            "max_length": self.profile.max_length,
            "loaded": self.is_loaded,
            "supports_chat": self.supports_chat,
            "memory_requirement": f"{self.profile.memory_gb}GB",
            "description": self.profile.description
        }
        
        if not self.is_loaded:
            return base_info
        
        # Calculate model size
        total_params = sum(p.numel() for p in self.model.parameters())
        model_size_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32
        
        base_info.update({
            "model_type": self.model.config.model_type if hasattr(self.model, 'config') else "Unknown",
            "model_size": f"~{model_size_mb:.1f}MB ({total_params:,} parameters)",
            "chat_template": self.profile.chat_template,
            "default_settings": {
                "temperature": self.profile.default_temperature,
                "top_p": self.profile.default_top_p,
                "top_k": self.profile.default_top_k,
                "max_tokens": self.profile.default_max_tokens
            }
        })
        
        return base_info
    
    def switch_model(self, model_name: str, progress_callback: Optional[Callable] = None) -> bool:
        """Switch to a different model with progress tracking"""
        if model_name == self.model_name:
            if progress_callback:
                progress = ModelLoadingProgress(self.profile.name)
                progress.update("complete", 100, f"Model {model_name} already loaded")
                progress_callback(progress)
            return True
        
        # Progress tracking for cleanup
        if progress_callback:
            progress = ModelLoadingProgress(f"Switching to {model_name}")
            progress.add_callback(progress_callback)
            progress.update("cleanup", 5, f"Unloading current model ({self.model_name})...")
        
        # Unload current model
        if self.is_loaded:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if progress_callback:
            progress.update("initializing", 10, f"Initializing {model_name}...")
        
        # Load new model
        self.model_name = model_name
        self.profile = get_model_profile(model_name)
        self.supports_chat = self.profile.supports_chat
        self.is_loaded = False
        
        return True