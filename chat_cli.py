#!/usr/bin/env python3
"""
Command-line chat interface for the LLM Serving API with Llama3 support
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.layout import Layout
from rich.align import Align
from prompt_toolkit import prompt as toolkit_prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.shortcuts import confirm

# Voice input support (optional)
try:
    from app.voice import VoiceInputManager, VoiceInputMode
    from app.core.config import settings
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    VoiceInputManager = None
    VoiceInputMode = None
    settings = None

# Configuration
DEFAULT_API_URL = "http://localhost:8000"
CONVERSATIONS_DIR = Path("./conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)

console = Console()


class ChatAPI:
    """API client for the LLM serving backend"""
    
    def __init__(self, base_url: str = DEFAULT_API_URL):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=60.0)
        self.conversation_id: Optional[str] = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def test_connection(self) -> bool:
        """Test if the API server is reachable"""
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/health")
            return response.status_code == 200
        except Exception:
            return False
    
    async def get_models(self) -> Dict[str, Any]:
        """Get available models"""
        response = await self.client.get(f"{self.base_url}/api/v1/chat/models")
        response.raise_for_status()
        return response.json()
    
    async def switch_model(self, model_name: str) -> Dict[str, Any]:
        """Switch to a different model"""
        response = await self.client.post(
            f"{self.base_url}/api/v1/chat/switch-model",
            json={"model_name": model_name}
        )
        response.raise_for_status()
        return response.json()
    
    async def new_conversation(self, system_prompt: str = None, model_name: str = None) -> Dict[str, Any]:
        """Start a new conversation"""
        data = {}
        if system_prompt:
            data["system_prompt"] = system_prompt
        if model_name:
            data["model_name"] = model_name
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/chat/new",
            json=data
        )
        response.raise_for_status()
        result = response.json()
        self.conversation_id = result["id"]
        return result
    
    async def send_message(
        self,
        message: str,
        model_name: str = None,
        max_tokens: int = 200,
        temperature: float = 0.6
    ) -> Dict[str, Any]:
        """Send a chat message"""
        data = {
            "message": message,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if self.conversation_id:
            data["conversation_id"] = self.conversation_id
        if model_name:
            data["model_name"] = model_name
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/chat",
            json=data
        )
        response.raise_for_status()
        result = response.json()
        
        # Update conversation ID if this was a new conversation
        if not self.conversation_id:
            self.conversation_id = result["conversation_id"]
        
        return result
    
    async def get_conversation(self, conversation_id: str = None) -> Dict[str, Any]:
        """Get conversation history"""
        conv_id = conversation_id or self.conversation_id
        if not conv_id:
            raise ValueError("No conversation ID available")
        
        response = await self.client.get(f"{self.base_url}/api/v1/chat/conversation/{conv_id}")
        response.raise_for_status()
        return response.json()


class ChatInterface:
    """Rich command-line chat interface with voice input support"""
    
    def __init__(self, api: ChatAPI):
        self.api = api
        self.history = InMemoryHistory()
        self.current_model = "gpt2"
        self.available_models = {}
        self.chat_models = []
        
        # Voice input support
        self.voice_manager = None
        self.voice_enabled = False
        self.voice_mode = False  # Toggle between text and voice input
        
        # Initialize voice input if available
        if VOICE_AVAILABLE:
            try:
                # Check if voice input is enabled in config
                voice_enabled_in_config = True
                if settings and hasattr(settings, 'voice'):
                    voice_enabled_in_config = settings.voice.enabled
                
                if not voice_enabled_in_config:
                    console.print("[dim]Voice input disabled in configuration[/dim]")
                    self.voice_enabled = False
                    return
                
                # Use settings from config if available
                voice_config = {}
                if settings and hasattr(settings, 'voice'):
                    voice_config = {
                        'whisper_model': settings.voice.whisper_model,
                        'language': settings.voice.language,
                        'sample_rate': settings.voice.sample_rate,
                        'silence_threshold': settings.voice.silence_threshold,
                        'silence_duration': settings.voice.silence_duration,
                        'max_recording_time': settings.voice.max_recording_time,
                        'device_index': settings.voice.device_index
                    }
                
                self.voice_manager = VoiceInputManager(**voice_config)
                self.voice_enabled = True
            except Exception as e:
                console.print(f"[yellow]⚠️ Voice input not available: {e}")
                self.voice_enabled = False
        
    async def initialize(self):
        """Initialize the chat interface"""
        # Test connection
        if not await self.api.test_connection():
            console.print("[red]❌ Cannot connect to API server at {self.api.base_url}")
            console.print(f"[yellow]💡 Make sure the server is running:")
            console.print(f"[cyan]   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
            return False
        
        # Get available models
        try:
            models_data = await self.api.get_models()
            self.available_models = {m["name"]: m for m in models_data["models"]}
            self.chat_models = models_data["chat_models"]
            self.current_model = models_data["current_model"]
        except Exception as e:
            console.print(f"[red]❌ Failed to get models: {e}")
            return False
        
        # Initialize voice input if enabled
        if self.voice_enabled and self.voice_manager:
            try:
                voice_ok = await self.voice_manager.initialize()
                if not voice_ok:
                    console.print(f"[yellow]⚠️ Voice input initialization failed, continuing without voice")
                    self.voice_enabled = False
                else:
                    console.print(f"[green]✅ Voice input ready (Whisper model: {self.voice_manager.transcriber.model_name})")
            except Exception as e:
                console.print(f"[yellow]⚠️ Voice input error: {e}")
                self.voice_enabled = False
        
        return True
    
    def show_welcome(self):
        """Display welcome message"""
        voice_status = ""
        voice_commands = ""
        
        if self.voice_enabled:
            voice_status = "\n[green]🎤 Voice input enabled[/green]"
            voice_commands = """• [cyan]/voice[/cyan] - Toggle voice input mode
• [cyan]/record[/cyan] - Record a voice message
• [cyan]/voice-settings[/cyan] - Voice configuration
"""
        
        welcome_text = f"""
[bold blue]🤖 LLM Chat Interface with Voice Support[/bold blue]
[dim]Connected to: {self.api.base_url}[/dim]{voice_status}

[bold]Available Commands:[/bold]
• [cyan]/help[/cyan] - Show this help
• [cyan]/models[/cyan] - List available models
• [cyan]/switch <model>[/cyan] - Switch models
{voice_commands}• [cyan]/clear[/cyan] - Clear conversation history  
• [cyan]/save <filename>[/cyan] - Save conversation
• [cyan]/load <filename>[/cyan] - Load conversation
• [cyan]/quit[/cyan] - Exit

[bold]Current Model:[/bold] [green]{self.current_model}[/green]
[bold]Chat Models:[/bold] {', '.join(self.chat_models)}

[dim]Type your message and press Enter to chat!{' Or use /voice to enable voice input!' if self.voice_enabled else ''}[/dim]
        """
        
        console.print(Panel(welcome_text, title="🚀 Welcome", border_style="blue"))
    
    def show_models(self):
        """Display available models in a table"""
        table = Table(title="📚 Available Models")
        table.add_column("Name", style="cyan")
        table.add_column("Display Name", style="green")
        table.add_column("Chat Support", style="yellow")
        table.add_column("Memory", style="magenta")
        table.add_column("Description", style="white")
        
        for name, model in self.available_models.items():
            chat_support = "✅" if model["supports_chat"] else "❌"
            is_current = "👑 " if name == self.current_model else ""
            table.add_row(
                f"{is_current}{name}",
                model["display_name"],
                chat_support,
                model["memory_requirement"],
                model["description"]
            )
        
        console.print(table)
    
    async def switch_model(self, model_name: str):
        """Switch to a different model with enhanced progress tracking"""
        if model_name not in self.available_models:
            console.print(f"[red]❌ Unknown model: {model_name}")
            console.print(f"[yellow]Available models: {', '.join(self.available_models.keys())}")
            return
        
        if model_name == self.current_model:
            console.print(f"[yellow]Already using model: {model_name}")
            return
        
        model_info = self.available_models[model_name]
        memory_req = model_info.get("memory_requirement", "Unknown")
        
        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
            
            # Show model info before switching
            console.print(f"\n[bold blue]🔄 Switching to: {model_name}[/bold blue]")
            console.print(f"[dim]Display Name: {model_info['display_name']}[/dim]")
            console.print(f"[dim]Memory Required: {memory_req}[/dim]")
            console.print(f"[dim]Description: {model_info['description']}[/dim]")
            
            # Create a detailed progress display
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.fields[stage]}"),
                BarColumn(bar_width=30, style="blue", complete_style="green"),
                TaskProgressColumn(),
                TextColumn("•"),
                TextColumn("{task.fields[status]}"),
                TimeElapsedColumn(),
                console=console,
                transient=False
            ) as progress:
                
                # Add the main progress task
                task_id = progress.add_task(
                    "Loading model...",
                    total=100,
                    stage="🔄 Initializing",
                    status="Starting model switch..."
                )
                
                # Start the API call
                import asyncio
                
                # Use a longer timeout for large models
                timeout = 300.0 if "GB" in memory_req and any(size in memory_req for size in ["10", "16", "24", "28"]) else 120.0
                
                async def make_switch_request():
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.post(
                            f"{self.api.base_url}/api/v1/chat/switch-model",
                            json={"model_name": model_name}
                        )
                        response.raise_for_status()
                        return response.json()
                
                # Simulate progress tracking since we don't have streaming yet
                # This provides user feedback during the potentially long loading process
                async def simulate_progress():
                    stages = [
                        (10, "🔧 Preparing", "Validating model configuration..."),
                        (20, "📥 Downloading", "Downloading model files (first time only)..."),
                        (40, "🔍 Loading", "Loading tokenizer and configuration..."),
                        (60, "🧠 Processing", "Loading model weights into memory..."),
                        (80, "⚙️ Optimizing", "Optimizing model for your hardware..."),
                        (95, "✅ Finalizing", "Almost ready...")
                    ]
                    
                    api_task = asyncio.create_task(make_switch_request())
                    
                    # Update progress while API call is running
                    stage_index = 0
                    last_progress = 5
                    
                    while not api_task.done():
                        await asyncio.sleep(1.0)  # Update every second
                        
                        # Move through stages progressively
                        if stage_index < len(stages):
                            target_progress, stage_name, stage_message = stages[stage_index]
                            
                            # Gradually increase progress toward target
                            if last_progress < target_progress:
                                last_progress = min(target_progress, last_progress + 2)
                                progress.update(
                                    task_id,
                                    completed=last_progress,
                                    stage=stage_name,
                                    status=stage_message
                                )
                                
                                # Move to next stage when target reached
                                if last_progress >= target_progress:
                                    stage_index += 1
                        else:
                            # Final stage - hold at 95% until completion
                            progress.update(
                                task_id,
                                completed=95,
                                stage="⏳ Loading",
                                status="Finalizing model loading, please wait..."
                            )
                    
                    # API call completed
                    result = await api_task
                    
                    # Complete the progress
                    progress.update(
                        task_id,
                        completed=100,
                        stage="✅ Complete",
                        status=f"Model loaded successfully!"
                    )
                    
                    return result
                
                # Run the progress simulation
                result = await simulate_progress()
            
            self.current_model = model_name
            load_time = result.get('load_time', 0)
            
            # Show success message with details
            console.print(f"\n[bold green]✅ Successfully switched to {result['model_name']}[/bold green]")
            console.print(f"[dim]Load time: {load_time:.1f}s | Memory: {memory_req} | Chat support: {'✅' if model_info['supports_chat'] else '❌'}[/dim]")
            
            # Start new conversation with new model
            await self.api.new_conversation()
            console.print("[dim]Started new conversation with new model[/dim]")
            
        except httpx.TimeoutException:
            console.print(f"\n[red]❌ Model loading timed out[/red]")
            console.print("[yellow]💡 Large models can take several minutes to download on first use")
            console.print("[yellow]   Please check your internet connection and try again")
            
        except Exception as e:
            console.print(f"\n[red]❌ Failed to switch model: {e}[/red]")
            
            # Provide helpful error information
            if "503" in str(e):
                console.print("[yellow]💡 Server might be starting up or busy. Please wait and try again.")
            elif "No GPU" in str(e) or "FP8" in str(e):
                console.print("[yellow]💡 This model requires GPU support. Try CPU-compatible alternatives:")
                cpu_models = [name for name, info in self.available_models.items() 
                             if "cpu compatible" in info.get("description", "").lower() or 
                                name.startswith(("gpt2", "qwen", "deepseek-coder"))]
                if cpu_models:
                    console.print(f"[cyan]   Recommended: {', '.join(cpu_models[:3])}[/cyan]")
            elif "memory" in str(e).lower() or "oom" in str(e).lower():
                console.print(f"[yellow]💡 Model requires {memory_req} RAM. Try a smaller model:")
                small_models = [name for name, info in self.available_models.items() 
                               if "1.8b" in name or "1.3b" in name or name.startswith("gpt2")]
                if small_models:
                    console.print(f"[cyan]   Try: {', '.join(small_models[:3])}[/cyan]")
    
    async def send_message(self, message: str):
        """Send a chat message and display response"""
        try:
            # Show typing indicator
            with Live(Spinner("dots", text="[dim]AI is thinking..."), console=console, transient=True):
                result = await self.api.send_message(message)
            
            # Display response
            response_text = result["response"]
            model_name = result["model_name"]
            generation_time = result["generation_time"]
            
            # Format response
            response_panel = Panel(
                response_text,
                title=f"🤖 {model_name}",
                subtitle=f"⏱️ {generation_time:.2f}s | 💬 {result['message_count']} messages",
                border_style="green"
            )
            
            console.print(response_panel)
            
        except Exception as e:
            console.print(f"[red]❌ Chat error: {e}")
    
    async def save_conversation(self, filename: str = None):
        """Save current conversation to file"""
        if not self.api.conversation_id:
            console.print("[yellow]No active conversation to save")
            return
        
        try:
            conversation = await self.api.get_conversation()
            
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"chat_{timestamp}.json"
            
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = CONVERSATIONS_DIR / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, indent=2, ensure_ascii=False, default=str)
            
            console.print(f"[green]✅ Conversation saved to {filepath}")
            
        except Exception as e:
            console.print(f"[red]❌ Save failed: {e}")
    
    def load_conversation(self, filename: str):
        """Load conversation from file"""
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = CONVERSATIONS_DIR / filename
        
        if not filepath.exists():
            console.print(f"[red]❌ File not found: {filepath}")
            # Show available files
            files = list(CONVERSATIONS_DIR.glob("*.json"))
            if files:
                console.print(f"[yellow]Available files: {', '.join(f.name for f in files)}")
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                conversation = json.load(f)
            
            console.print(f"[green]✅ Loaded conversation: {conversation['title']}")
            console.print(f"[dim]Created: {conversation['created_at']} | Messages: {sum(conversation['message_count'].values())}[/dim]")
            
            # Display conversation history
            self.display_conversation_history(conversation)
            
        except Exception as e:
            console.print(f"[red]❌ Load failed: {e}")
    
    def display_conversation_history(self, conversation: Dict[str, Any]):
        """Display conversation history"""
        console.print(Panel(f"📜 Conversation History: {conversation['title']}", border_style="blue"))
        
        for msg in conversation["messages"]:
            role = msg["role"]
            content = msg["content"]
            timestamp = msg.get("timestamp", "")
            
            if role == "system":
                continue  # Skip system messages in display
            elif role == "user":
                console.print(f"[bold cyan]👤 You:[/bold cyan] {content}")
            elif role == "assistant":
                console.print(f"[bold green]🤖 Assistant:[/bold green] {content}")
            
            console.print()
    
    async def clear_conversation(self):
        """Clear current conversation"""
        if confirm("Clear current conversation?"):
            await self.api.new_conversation()
            console.print("[green]✅ Conversation cleared")
    
    async def get_user_input(self) -> Optional[str]:
        """Unified method to get user input (text or voice)"""
        if self.voice_mode and self.voice_enabled:
            return await self.get_voice_input()
        else:
            return await self.get_text_input()
    
    async def get_text_input(self) -> str:
        """Get text input from user"""
        prompt_text = "💬 You: "
        if self.voice_enabled:
            mode_indicator = " 🎤" if self.voice_mode else " ⌨️"
            prompt_text = f"💬 You{mode_indicator}: "
        
        return toolkit_prompt(
            prompt_text,
            history=self.history,
            auto_suggest=AutoSuggestFromHistory()
        ).strip()
    
    async def get_voice_input(self) -> Optional[str]:
        """Get voice input from user with real-time feedback"""
        if not self.voice_enabled or not self.voice_manager:
            console.print("[red]❌ Voice input not available")
            return None
        
        try:
            # Show recording interface with enhanced feedback
            console.print("[bold green]🎤 Recording... (speak now, will auto-stop on silence)[/bold green]")
            console.print("[dim]Press Ctrl+C to cancel recording[/dim]")
            
            # Create a more interactive recording display
            from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
            
            with Progress(
                TextColumn("[bold blue]🎤 Recording"),
                BarColumn(bar_width=40, style="green", complete_style="green"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console,
                transient=True
            ) as progress:
                
                # Add progress task
                recording_task = progress.add_task(
                    "Recording audio...", 
                    total=self.voice_manager.max_recording_time
                )
                
                # Start recording with progress updates
                start_time = time.time()
                
                # Create a custom recording method with progress
                recording_future = asyncio.create_task(
                    self.voice_manager.get_voice_input(mode=VoiceInputMode.AUTO_STOP)
                )
                
                # Update progress while recording
                while not recording_future.done():
                    elapsed = time.time() - start_time
                    progress.update(recording_task, completed=elapsed)
                    
                    # Show audio level if available
                    try:
                        level = self.voice_manager.get_current_audio_level()
                        level_bars = self._get_audio_level_bars(level)
                        progress.update(recording_task, description=f"🎤 Recording {level_bars}")
                    except:
                        pass
                    
                    await asyncio.sleep(0.1)
                
                text = await recording_future
            
            if text:
                console.print(f"[dim]📝 Transcribed: {text}[/dim]")
                return text
            else:
                console.print("[yellow]⚠️ No speech detected or transcription failed")
                return None
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Recording cancelled[/yellow]")
            self.voice_manager.stop_recording()
            return None
        except Exception as e:
            console.print(f"[red]❌ Voice input error: {e}")
            return None
    
    def _get_audio_level_bars(self, level: float) -> str:
        """Create audio level visualization bars"""
        # Normalize level to 0-1 range
        level = max(0.0, min(1.0, level))
        
        # Create bar visualization
        bar_length = 10
        filled_bars = int(level * bar_length)
        empty_bars = bar_length - filled_bars
        
        if level > 0.7:
            color = "red"
        elif level > 0.4:
            color = "yellow"
        else:
            color = "green"
        
        bars = f"[{color}]{'█' * filled_bars}[/]{color}]{'░' * empty_bars}"
        return f"{bars} {level*100:.0f}%"
    
    async def toggle_voice_mode(self):
        """Toggle between text and voice input modes"""
        if not self.voice_enabled:
            console.print("[red]❌ Voice input not available")
            return
        
        self.voice_mode = not self.voice_mode
        mode = "voice" if self.voice_mode else "text"
        icon = "🎤" if self.voice_mode else "⌨️"
        console.print(f"[green]✅ Switched to {mode} input mode {icon}")
    
    async def record_voice_message(self):
        """Record a single voice message and send it"""
        if not self.voice_enabled:
            console.print("[red]❌ Voice input not available")
            return
        
        try:
            console.print("[bold green]🎤 Recording voice message... (speak now, will auto-stop on silence)[/bold green]")
            console.print("[dim]Press Ctrl+C to cancel recording[/dim]")
            
            # Use the same enhanced recording interface
            from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
            
            with Progress(
                TextColumn("[bold blue]🎤 Voice Message"),
                BarColumn(bar_width=40, style="green", complete_style="green"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console,
                transient=True
            ) as progress:
                
                recording_task = progress.add_task(
                    "Recording...", 
                    total=self.voice_manager.max_recording_time
                )
                
                start_time = time.time()
                recording_future = asyncio.create_task(
                    self.voice_manager.get_voice_input(mode=VoiceInputMode.AUTO_STOP)
                )
                
                while not recording_future.done():
                    elapsed = time.time() - start_time
                    progress.update(recording_task, completed=elapsed)
                    
                    try:
                        level = self.voice_manager.get_current_audio_level()
                        level_bars = self._get_audio_level_bars(level)
                        progress.update(recording_task, description=f"🎤 Voice Message {level_bars}")
                    except:
                        pass
                    
                    await asyncio.sleep(0.1)
                
                text = await recording_future
            
            if text:
                console.print(f"[dim]📝 Transcribed: {text}[/dim]")
                await self.send_message(text)
            else:
                console.print("[yellow]⚠️ No speech detected or transcription failed")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Recording cancelled[/yellow]")
            self.voice_manager.stop_recording()
        except Exception as e:
            console.print(f"[red]❌ Voice recording error: {e}")
    
    def show_voice_settings(self):
        """Display voice input settings"""
        if not self.voice_enabled:
            console.print("[red]❌ Voice input not available")
            return
        
        status = self.voice_manager.get_status()
        
        # Create a more detailed status display
        whisper_models = self.voice_manager.get_whisper_models()
        current_model = status.get('whisper_model', 'N/A')
        model_description = whisper_models.get(current_model, 'Unknown model')
        
        devices = self.voice_manager.get_audio_devices()
        current_device_index = status.get('device_index')
        current_device_name = "Default"
        
        if current_device_index is not None:
            for device in devices:
                if device['index'] == current_device_index:
                    current_device_name = device['name']
                    break
        
        # Voice mode status
        mode_status = "[green]Voice mode[/green]" if self.voice_mode else "[dim]Text mode[/dim]"
        
        settings_text = f"""
[bold]🎤 Voice Input Settings[/bold]

[bold]Current Status:[/bold]
• Input mode: {mode_status}
• Voice input: [green]{'Enabled' if status['is_enabled'] else 'Disabled'}[/green]
• Recording: [red]{'Active' if status['is_recording'] else 'Idle'}[/red]

[bold]Whisper Configuration:[/bold]
• Model: [cyan]{current_model}[/cyan] - {model_description}
• Language: [cyan]{status.get('language', 'auto-detect')}[/cyan]
• Model loaded: [green]{'Yes' if status.get('whisper_loaded') else 'No'}[/green]

[bold]Audio Configuration:[/bold]
• Device: [cyan]{current_device_name}[/cyan] (Index: {current_device_index or 'default'})
• Available devices: [cyan]{status.get('audio_devices_count', 0)}[/cyan]
• Sample rate: [cyan]16 kHz[/cyan]
• Channels: [cyan]Mono[/cyan]

[bold]Recording Settings:[/bold]
• Auto-stop on silence: [cyan]{settings.voice.silence_duration}s[/cyan]
• Max recording time: [cyan]{settings.voice.max_recording_time}s[/cyan]
• Silence threshold: [cyan]{settings.voice.silence_threshold}[/cyan]

[bold]Available Commands:[/bold]
• [cyan]/voice[/cyan] - Toggle voice input mode
• [cyan]/record[/cyan] - Record a single voice message
• [cyan]/devices[/cyan] - List audio devices
• [cyan]/voice-settings[/cyan] - Show this settings panel

[bold]💡 Voice Input Tips:[/bold]
• Speak clearly after the recording starts
• Voice mode auto-stops on silence
• All transcription happens locally (offline)
• Use /devices to see available microphones
• Try different Whisper models for better accuracy
        """
        
        console.print(Panel(settings_text, title="🎤 Voice Settings", border_style="green"))
    
    def show_audio_devices(self):
        """Display available audio input devices"""
        if not self.voice_enabled:
            console.print("[red]❌ Voice input not available")
            return
        
        devices = self.voice_manager.get_audio_devices()
        
        if not devices:
            console.print("[yellow]⚠️ No audio input devices found")
            return
        
        table = Table(title="🎤 Audio Input Devices")
        table.add_column("Index", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Channels", style="yellow")
        table.add_column("Sample Rate", style="magenta")
        table.add_column("Status", style="white")
        
        current_device = self.voice_manager.device_index
        
        for device in devices:
            status = "👑 Current" if device['index'] == current_device else ""
            table.add_row(
                str(device['index']),
                device['name'],
                str(device['channels']),
                f"{device['sample_rate']:.0f} Hz",
                status
            )
        
        console.print(table)
    
    def get_voice_status_panel(self) -> Optional[Panel]:
        """Create a voice status panel for the interface"""
        if not self.voice_enabled:
            return None
        
        try:
            status = self.voice_manager.get_status()
            
            # Create status indicators
            mode_icon = "🎤" if self.voice_mode else "⌨️"
            mode_text = "Voice" if self.voice_mode else "Text"
            recording_status = "🔴 Recording" if status.get('is_recording') else "⚪ Ready"
            
            whisper_status = "✅ Loaded" if status.get('whisper_loaded') else "⏳ Loading"
            
            status_text = f"{mode_icon} {mode_text} | {recording_status} | Whisper: {whisper_status}"
            
            return Panel(
                status_text,
                title="🎤 Voice Status",
                border_style="dim",
                padding=(0, 1)
            )
        except Exception:
            return None
    
    def show_help(self):
        """Show help message"""
        # Base commands
        base_commands = """[cyan]/help[/cyan] - Show this help message
[cyan]/models[/cyan] - List all available models
[cyan]/switch <model>[/cyan] - Switch to a different model
[cyan]/clear[/cyan] - Clear conversation history
[cyan]/save [filename][/cyan] - Save conversation (auto-named if no filename)
[cyan]/load <filename>[/cyan] - Load saved conversation
[cyan]/quit[/cyan] or [cyan]/exit[/cyan] - Exit the chat"""
        
        # Voice commands (if available)
        voice_commands = ""
        voice_tips = ""
        
        if self.voice_enabled:
            voice_commands = """
[bold]🎤 Voice Commands:[/bold]
[cyan]/voice[/cyan] - Toggle voice input mode
[cyan]/record[/cyan] - Record a single voice message
[cyan]/voice-settings[/cyan] - Show voice configuration
[cyan]/devices[/cyan] - List audio input devices
"""
            voice_tips = """• Use /voice to toggle between text and voice input
• Voice input auto-stops on silence detection
• All transcription happens locally (offline)
• """
        
        help_text = f"""
[bold]📚 Available Commands:[/bold]

{base_commands}{voice_commands}

[bold]💡 Tips:[/bold]
• Use Tab for auto-completion
• Press Ctrl+C to interrupt generation
• Conversations auto-save every 10 messages
• Model switching shows detailed progress for large downloads
• Use Qwen and DeepSeek models for best performance
{voice_tips}
[bold]🦙 Recommended Models:[/bold]
• [green]qwen3-1.8b[/green] - High-quality multilingual, 6GB RAM (default)
• [green]qwen3-3b[/green] - Advanced quality, 8GB RAM required
• [green]deepseek-coder-1.3b[/green] - Excellent for coding tasks, 6GB RAM
• [green]deepseek-coder-6.7b[/green] - Advanced coding assistance, 16GB RAM
• [green]deepseek-math-7b[/green] - Mathematical reasoning, 16GB RAM
• [green]gpt2[/green] - Fast lightweight option, 2GB RAM

[bold]⚠️ MacBook Pro Notes:[/bold]
• All models are optimized for CPU/MPS (Metal Performance Shaders)
• First-time downloads may take several minutes for large models
• Models are cached locally after first download
        """
        console.print(Panel(help_text, title="🆘 Help", border_style="blue"))
    
    async def run(self):
        """Main chat loop"""
        if not await self.initialize():
            return
        
        self.show_welcome()
        
        # Start new conversation
        await self.api.new_conversation()
        
        console.print("\n[bold green]Ready to chat! Type your message or /help for commands.[/bold green]\n")
        
        while True:
            try:
                # Get user input (text or voice)
                user_input = await self.get_user_input()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    await self.handle_command(user_input)
                else:
                    # Send chat message
                    await self.send_message(user_input)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]⚠️ Interrupted. Type /quit to exit.[/yellow]")
                continue
            except EOFError:
                break
        
        console.print("\n[blue]👋 Goodbye![/blue]")
    
    async def handle_command(self, command: str):
        """Handle user commands"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd in ['/quit', '/exit']:
            if confirm("Really exit?"):
                sys.exit(0)
        
        elif cmd == '/help':
            self.show_help()
        
        elif cmd == '/models':
            self.show_models()
        
        elif cmd == '/switch':
            if len(parts) < 2:
                console.print("[red]Usage: /switch <model_name>")
                self.show_models()
            else:
                await self.switch_model(parts[1])
        
        elif cmd == '/clear':
            await self.clear_conversation()
        
        elif cmd == '/save':
            filename = parts[1] if len(parts) > 1 else None
            await self.save_conversation(filename)
        
        elif cmd == '/load':
            if len(parts) < 2:
                console.print("[red]Usage: /load <filename>")
                files = list(CONVERSATIONS_DIR.glob("*.json"))
                if files:
                    console.print(f"[yellow]Available files: {', '.join(f.name for f in files)}")
            else:
                self.load_conversation(parts[1])
        
        # Voice commands
        elif cmd == '/voice':
            await self.toggle_voice_mode()
        
        elif cmd == '/record':
            await self.record_voice_message()
        
        elif cmd == '/voice-settings':
            self.show_voice_settings()
        
        elif cmd == '/devices':
            self.show_audio_devices()
        
        else:
            console.print(f"[red]Unknown command: {command}")
            console.print("[yellow]Type /help for available commands")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LLM Chat Interface with Llama3")
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"API server URL (default: {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--model",
        help="Initial model to use"
    )
    
    args = parser.parse_args()
    
    async with ChatAPI(args.api_url) as api:
        chat = ChatInterface(api)
        
        # Switch to requested model if specified
        if args.model:
            await chat.initialize()
            await chat.switch_model(args.model)
        
        await chat.run()


def run_chat():
    """Main entry point that handles event loop issues"""
    # First check if we can import nest_asyncio to handle nested loops
    try:
        import nest_asyncio
        nest_asyncio.apply()
        console.print("[blue]🔧 Applied nest_asyncio for compatibility")
    except ImportError:
        pass
    
    # Try to run the chat
    try:
        asyncio.run(main())
        return True
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            console.print("[red]❌ Cannot start chat interface from within an async environment")
            console.print("")
            console.print("[yellow]💡 Solutions:")
            console.print("[cyan]   1. Run from a regular terminal/command prompt:")
            console.print("[cyan]      python chat_cli.py")
            console.print("")
            console.print("[cyan]   2. Or install nest-asyncio and try again:")
            console.print("[cyan]      pip install nest-asyncio")
            console.print("")
            console.print("[cyan]   3. Or use the API directly with curl:")
            console.print("[cyan]      curl -X POST http://localhost:8000/api/v1/chat/new")
            console.print("")
            console.print("[cyan]   4. Or use the web interface:")
            console.print("[cyan]      http://localhost:8000/docs")
            console.print("")
            console.print(f"[dim]Current directory: {Path.cwd()}")
            console.print(f"[dim]Error: {e}")
            return False
        else:
            # Re-raise other RuntimeErrors
            raise
    except Exception as e:
        console.print(f"[red]❌ Error starting chat: {e}")
        return False


if __name__ == "__main__":
    try:
        run_chat()
    except KeyboardInterrupt:
        console.print("\n[blue]👋 Goodbye![/blue]")
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}")
        console.print("[yellow]💡 Try running from a regular terminal")