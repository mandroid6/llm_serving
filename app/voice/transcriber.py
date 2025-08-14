"""
Whisper-based speech-to-text transcription
"""

import asyncio
import io
import logging
import os
import tempfile
import warnings
from typing import Optional, Dict, Any
import wave

try:
    import whisper
    import numpy as np
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

logger = logging.getLogger(__name__)

# Function to conditionally suppress warnings based on settings
def _maybe_suppress_warnings():
    """Suppress Whisper warnings based on configuration"""
    try:
        from app.core.config import settings
        if hasattr(settings, 'voice') and settings.voice.suppress_warnings:
            warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
            warnings.filterwarnings("ignore", message=".*torch.nn.utils.weight_norm.*")
            warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
    except ImportError:
        # Default to suppressing warnings if settings not available
        warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
        warnings.filterwarnings("ignore", message=".*torch.nn.utils.weight_norm.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

# Apply warning suppression
_maybe_suppress_warnings()


class WhisperTranscriber:
    """Handles speech-to-text transcription using OpenAI Whisper"""
    
    # Available Whisper models with their approximate sizes
    MODELS = {
        "tiny": "~39 MB - fastest, lowest accuracy",
        "base": "~74 MB - good balance of speed and accuracy", 
        "small": "~244 MB - better accuracy, slower",
        "medium": "~769 MB - high accuracy, much slower",
        "large": "~1550 MB - best accuracy, very slow"
    }
    
    def __init__(
        self,
        model_name: str = "base",
        language: Optional[str] = None,
        device: str = "cpu"
    ):
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "OpenAI Whisper is required for speech-to-text. Install it with: pip install openai-whisper"
            )
        
        self.model_name = model_name
        self.language = language  # None for auto-detection
        self.device = device
        self.model = None
        self._is_loading = False
        
        # Check if warnings should be suppressed
        self.suppress_warnings = True
        try:
            from app.core.config import settings
            if hasattr(settings, 'voice'):
                self.suppress_warnings = settings.voice.suppress_warnings
        except ImportError:
            pass  # Default to suppressing warnings
        
        # Validate model name
        if model_name not in self.MODELS:
            logger.warning(f"Unknown model '{model_name}', using 'base' instead")
            self.model_name = "base"
    
    async def load_model(self) -> bool:
        """Load the Whisper model"""
        if self.model is not None:
            return True
        
        if self._is_loading:
            # Wait for other loading process to complete
            while self._is_loading:
                await asyncio.sleep(0.1)
            return self.model is not None
        
        self._is_loading = True
        
        try:
            logger.info(f"Loading Whisper model '{self.model_name}' ({self.MODELS[self.model_name]})")
            
            # Load model in a thread to avoid blocking, with warnings suppressed
            loop = asyncio.get_event_loop()
            
            def load_model_with_suppressed_warnings():
                if self.suppress_warnings:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        return whisper.load_model(self.model_name, device=self.device)
                else:
                    return whisper.load_model(self.model_name, device=self.device)
            
            self.model = await loop.run_in_executor(
                None,
                load_model_with_suppressed_warnings
            )
            
            logger.info(f"Whisper model '{self.model_name}' loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False
        finally:
            self._is_loading = False
    
    def _save_audio_to_temp_file(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Save audio bytes to a temporary WAV file"""
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Write WAV file
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)
            
            return temp_path
            
        except Exception as e:
            # Clean up on error
            try:
                os.close(temp_fd)
                os.unlink(temp_path)
            except:
                pass
            raise e
        finally:
            try:
                os.close(temp_fd)
            except:
                pass
    
    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        **whisper_options
    ) -> Dict[str, Any]:
        """
        Transcribe audio data to text
        
        Returns:
            Dict with keys: 'text', 'language', 'confidence', 'segments'
        """
        if not self.model:
            if not await self.load_model():
                raise RuntimeError("Failed to load Whisper model")
        
        temp_path = None
        try:
            # Save audio to temporary file
            temp_path = self._save_audio_to_temp_file(audio_data, sample_rate)
            
            # Prepare transcription options
            options = {
                'language': self.language,
                'task': 'transcribe',
                **whisper_options
            }
            
            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}
            
            logger.info(f"Transcribing audio file: {temp_path}")
            
            # Run transcription in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Conditionally suppress warnings during transcription
            def transcribe_with_optional_warning_suppression():
                if self.suppress_warnings:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        return self.model.transcribe(temp_path, **options)
                else:
                    return self.model.transcribe(temp_path, **options)
            
            result = await loop.run_in_executor(
                None,
                transcribe_with_optional_warning_suppression
            )
            
            # Extract information
            text = result.get('text', '').strip()
            language = result.get('language', 'unknown')
            segments = result.get('segments', [])
            
            # Calculate confidence (average of segment confidences if available)
            confidence = 0.0
            if segments:
                confidences = [seg.get('avg_logprob', 0.0) for seg in segments]
                if confidences:
                    # Convert log probabilities to rough confidence scores
                    confidence = np.exp(np.mean(confidences)) if confidences else 0.0
            
            logger.info(f"Transcription completed: '{text[:50]}...' (language: {language})")
            
            return {
                'text': text,
                'language': language,
                'confidence': confidence,
                'segments': segments,
                'duration': result.get('duration', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_path}: {e}")
    
    async def transcribe_with_timestamps(
        self,
        audio_data: bytes,
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        Transcribe with word-level timestamps
        """
        return await self.transcribe(
            audio_data,
            sample_rate,
            word_timestamps=True
        )
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported language codes and names"""
        try:
            # Get languages from Whisper
            languages = whisper.tokenizer.LANGUAGES
            return languages
        except Exception as e:
            logger.error(f"Failed to get supported languages: {e}")
            return {}
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Get available Whisper models"""
        return cls.MODELS.copy()
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different Whisper model"""
        if model_name not in self.MODELS:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        if model_name == self.model_name:
            return True  # Already using this model
        
        # Clear current model
        self.model = None
        self.model_name = model_name
        
        logger.info(f"Switched to Whisper model: {model_name}")
        return True
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model"""
        return {
            'model_name': self.model_name,
            'description': self.MODELS.get(self.model_name, 'Unknown'),
            'is_loaded': self.is_model_loaded(),
            'language': self.language or 'auto-detect',
            'device': self.device
        }