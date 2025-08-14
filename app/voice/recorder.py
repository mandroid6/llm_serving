"""
Audio recording functionality for voice input
"""

import asyncio
import time
import threading
from typing import Optional, List, Tuple
import logging

try:
    import pyaudio
    import numpy as np
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None

logger = logging.getLogger(__name__)


class VoiceRecorder:
    """Handles audio recording with silence detection and cross-platform support"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        silence_threshold: float = 0.01,
        silence_duration: float = 2.0
    ):
        if not PYAUDIO_AVAILABLE:
            raise ImportError(
                "PyAudio is required for voice input. Install it with: pip install pyaudio"
            )
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size 
        self.channels = channels
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        
        self.audio = None
        self.stream = None
        self.is_recording = False
        self.audio_data = []
        self.recording_thread = None
        
        # Initialize PyAudio
        self._init_audio()
    
    def _init_audio(self):
        """Initialize PyAudio interface"""
        try:
            self.audio = pyaudio.PyAudio()
            logger.info("PyAudio initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PyAudio: {e}")
            raise
    
    def get_audio_devices(self) -> List[dict]:
        """Get list of available audio input devices"""
        devices = []
        if not self.audio:
            return devices
            
        try:
            device_count = self.audio.get_device_count()
            for i in range(device_count):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:  # Input device
                    devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': device_info['defaultSampleRate']
                    })
        except Exception as e:
            logger.error(f"Error getting audio devices: {e}")
            
        return devices
    
    def get_default_input_device(self) -> Optional[dict]:
        """Get the default input device"""
        try:
            default_device = self.audio.get_default_input_device_info()
            return {
                'index': default_device['index'],
                'name': default_device['name'],
                'channels': default_device['maxInputChannels'],
                'sample_rate': default_device['defaultSampleRate']
            }
        except Exception as e:
            logger.error(f"Error getting default input device: {e}")
            return None
    
    def _calculate_audio_level(self, audio_chunk: bytes) -> float:
        """Calculate audio level from chunk for visualization"""
        try:
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            level = np.abs(audio_array).mean() / 32768.0  # Normalize to 0-1
            return min(level, 1.0)
        except Exception:
            return 0.0
    
    def _is_silence(self, audio_chunk: bytes) -> bool:
        """Check if audio chunk contains silence"""
        level = self._calculate_audio_level(audio_chunk)
        return level < self.silence_threshold
    
    async def record_until_silence(
        self, 
        max_duration: int = 60,
        device_index: Optional[int] = None
    ) -> Tuple[bytes, float]:
        """
        Record audio until silence is detected
        Returns (audio_data, duration)
        """
        if self.is_recording:
            raise RuntimeError("Already recording")
        
        self.is_recording = True
        self.audio_data = []
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )
            
            logger.info("Recording started...")
            start_time = time.time()
            silence_start = None
            
            while self.is_recording and (time.time() - start_time) < max_duration:
                try:
                    # Read audio chunk
                    chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    self.audio_data.append(chunk)
                    
                    # Check for silence
                    if self._is_silence(chunk):
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start >= self.silence_duration:
                            logger.info("Silence detected, stopping recording")
                            break
                    else:
                        silence_start = None  # Reset silence timer
                        
                    # Small delay to prevent busy waiting
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"Error reading audio: {e}")
                    break
            
            duration = time.time() - start_time
            audio_bytes = b''.join(self.audio_data)
            
            logger.info(f"Recording finished. Duration: {duration:.2f}s, Size: {len(audio_bytes)} bytes")
            return audio_bytes, duration
            
        except Exception as e:
            logger.error(f"Recording error: {e}")
            raise
        finally:
            self._stop_recording()
    
    async def record_manual_stop(
        self, 
        max_duration: int = 60,
        device_index: Optional[int] = None
    ) -> Tuple[bytes, float]:
        """
        Record audio until manually stopped
        Returns (audio_data, duration)
        """
        if self.is_recording:
            raise RuntimeError("Already recording")
        
        self.is_recording = True
        self.audio_data = []
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )
            
            logger.info("Recording started (manual stop mode)...")
            start_time = time.time()
            
            while self.is_recording and (time.time() - start_time) < max_duration:
                try:
                    chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    self.audio_data.append(chunk)
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"Error reading audio: {e}")
                    break
            
            duration = time.time() - start_time
            audio_bytes = b''.join(self.audio_data)
            
            logger.info(f"Recording finished. Duration: {duration:.2f}s, Size: {len(audio_bytes)} bytes")
            return audio_bytes, duration
            
        except Exception as e:
            logger.error(f"Recording error: {e}")
            raise
        finally:
            self._stop_recording()
    
    def stop_recording(self):
        """Stop current recording"""
        self.is_recording = False
    
    def _stop_recording(self):
        """Internal method to clean up recording resources"""
        self.is_recording = False
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
            finally:
                self.stream = None
    
    def get_current_audio_level(self) -> float:
        """Get current audio level (0.0 to 1.0) for visualization"""
        if not self.audio_data:
            return 0.0
        
        # Use the most recent chunk
        try:
            latest_chunk = self.audio_data[-1] if self.audio_data else b''
            return self._calculate_audio_level(latest_chunk)
        except Exception:
            return 0.0
    
    def cleanup(self):
        """Clean up audio resources"""
        self._stop_recording()
        
        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")
            finally:
                self.audio = None
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()