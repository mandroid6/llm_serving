"""
Main voice input manager that orchestrates recording and transcription
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Tuple
from enum import Enum

from .recorder import VoiceRecorder
from .transcriber import WhisperTranscriber

logger = logging.getLogger(__name__)


class VoiceInputMode(Enum):
    """Voice input modes"""
    AUTO_STOP = "auto_stop"        # Stop on silence
    MANUAL_STOP = "manual_stop"    # Stop when user presses key
    PUSH_TO_TALK = "push_to_talk"  # Hold key while talking


class VoiceInputManager:
    """Main manager for voice input functionality"""
    
    def __init__(
        self,
        whisper_model: str = "base",
        language: Optional[str] = None,
        sample_rate: int = 16000,
        silence_threshold: float = 0.01,
        silence_duration: float = 2.0,
        max_recording_time: int = 60,
        device_index: Optional[int] = None
    ):
        # Initialize components
        self.recorder = VoiceRecorder(
            sample_rate=sample_rate,
            silence_threshold=silence_threshold,
            silence_duration=silence_duration
        )
        
        self.transcriber = WhisperTranscriber(
            model_name=whisper_model,
            language=language
        )
        
        # Configuration
        self.max_recording_time = max_recording_time
        self.device_index = device_index
        self.is_enabled = True
        
        # State
        self.is_recording = False
        self._current_mode = VoiceInputMode.AUTO_STOP
    
    async def initialize(self) -> bool:
        """Initialize voice input system"""
        try:
            # Test audio devices
            devices = self.recorder.get_audio_devices()
            if not devices:
                logger.error("No audio input devices found")
                return False
            
            # Load Whisper model
            if not await self.transcriber.load_model():
                logger.error("Failed to load Whisper model")
                return False
            
            logger.info("Voice input system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize voice input: {e}")
            return False
    
    async def get_voice_input(
        self,
        mode: VoiceInputMode = VoiceInputMode.AUTO_STOP,
        max_duration: Optional[int] = None
    ) -> Optional[str]:
        """
        Get voice input and return transcribed text
        
        Args:
            mode: Voice input mode
            max_duration: Maximum recording duration (uses default if None)
            
        Returns:
            Transcribed text or None if failed/cancelled
        """
        if not self.is_enabled:
            logger.warning("Voice input is disabled")
            return None
        
        if self.is_recording:
            logger.warning("Already recording")
            return None
        
        max_duration = max_duration or self.max_recording_time
        self._current_mode = mode
        
        try:
            # Record audio based on mode
            if mode == VoiceInputMode.AUTO_STOP:
                audio_data, duration = await self.recorder.record_until_silence(
                    max_duration=max_duration,
                    device_index=self.device_index
                )
            elif mode == VoiceInputMode.MANUAL_STOP:
                audio_data, duration = await self.recorder.record_manual_stop(
                    max_duration=max_duration,
                    device_index=self.device_index
                )
            else:
                logger.error(f"Unsupported voice input mode: {mode}")
                return None
            
            # Check if we got any audio
            if not audio_data or len(audio_data) == 0:
                logger.warning("No audio data recorded")
                return None
            
            logger.info(f"Recorded {duration:.2f}s of audio, transcribing...")
            
            # Transcribe audio
            result = await self.transcriber.transcribe(
                audio_data,
                sample_rate=self.recorder.sample_rate
            )
            
            text = result.get('text', '').strip()
            language = result.get('language', 'unknown')
            confidence = result.get('confidence', 0.0)
            
            if not text:
                logger.warning("No text transcribed from audio")
                return None
            
            logger.info(f"Transcription: '{text}' (language: {language}, confidence: {confidence:.2f})")
            return text
            
        except Exception as e:
            logger.error(f"Voice input failed: {e}")
            return None
        finally:
            self.is_recording = False
    
    def start_recording(self):
        """Start recording (for manual control)"""
        # This will be handled by the specific recording methods
        self.is_recording = True
    
    def stop_recording(self):
        """Stop current recording"""
        self.recorder.stop_recording()
        self.is_recording = False
    
    def get_audio_devices(self) -> list:
        """Get available audio input devices"""
        return self.recorder.get_audio_devices()
    
    def get_default_device(self) -> Optional[dict]:
        """Get default audio input device"""
        return self.recorder.get_default_input_device()
    
    def set_audio_device(self, device_index: int) -> bool:
        """Set audio input device"""
        devices = self.get_audio_devices()
        if any(d['index'] == device_index for d in devices):
            self.device_index = device_index
            logger.info(f"Set audio device to index: {device_index}")
            return True
        else:
            logger.error(f"Invalid device index: {device_index}")
            return False
    
    def get_whisper_models(self) -> Dict[str, str]:
        """Get available Whisper models"""
        return self.transcriber.get_available_models()
    
    def switch_whisper_model(self, model_name: str) -> bool:
        """Switch to different Whisper model"""
        return self.transcriber.switch_model(model_name)
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages"""
        return self.transcriber.get_supported_languages()
    
    def set_language(self, language: Optional[str]):
        """Set transcription language (None for auto-detect)"""
        self.transcriber.language = language
        logger.info(f"Set transcription language to: {language or 'auto-detect'}")
    
    def get_current_audio_level(self) -> float:
        """Get current audio level for visualization"""
        return self.recorder.get_current_audio_level()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            'is_enabled': self.is_enabled,
            'is_recording': self.is_recording,
            'current_mode': self._current_mode.value,
            'whisper_model': self.transcriber.model_name,
            'whisper_loaded': self.transcriber.is_model_loaded(),
            'language': self.transcriber.language or 'auto-detect',
            'device_index': self.device_index,
            'audio_devices_count': len(self.get_audio_devices())
        }
    
    def enable(self):
        """Enable voice input"""
        self.is_enabled = True
        logger.info("Voice input enabled")
    
    def disable(self):
        """Disable voice input"""
        self.is_enabled = False
        if self.is_recording:
            self.stop_recording()
        logger.info("Voice input disabled")
    
    def cleanup(self):
        """Clean up resources"""
        if self.is_recording:
            self.stop_recording()
        self.recorder.cleanup()
        logger.info("Voice input manager cleaned up")
    
    async def test_voice_input(self) -> bool:
        """Test voice input functionality"""
        try:
            logger.info("Testing voice input...")
            
            # Test audio devices
            devices = self.get_audio_devices()
            if not devices:
                logger.error("No audio devices available")
                return False
            
            # Test Whisper model
            if not self.transcriber.is_model_loaded():
                if not await self.transcriber.load_model():
                    logger.error("Failed to load Whisper model")
                    return False
            
            logger.info("Voice input test passed")
            return True
            
        except Exception as e:
            logger.error(f"Voice input test failed: {e}")
            return False