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
import threading

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

# Enhanced keyboard handling for voice mode
try:
    from pynput import keyboard as pynput_keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

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


class VoiceExitHandler:
    """Handles graceful exit from voice mode using Esc key (no admin privileges required)"""
    
    def __init__(self):
        self.exit_requested = False
        self.monitoring = False
        self._stop_event = threading.Event()
        self._listener = None
        
    def start_monitoring(self):
        """Start monitoring for Esc key press"""
        self.exit_requested = False
        self.monitoring = True
        self._stop_event.clear()
        
        if PYNPUT_AVAILABLE:
            # Use pynput library if available (works without admin privileges)
            self._start_pynput_monitoring()
        else:
            # Fallback: use a simple flag-based approach
            self._start_simple_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring for key presses"""
        self.monitoring = False
        self._stop_event.set()
        
        if self._listener:
            try:
                self._listener.stop()
                self._listener = None
            except:
                pass
    
    def _start_pynput_monitoring(self):
        """Monitor using pynput library (no admin privileges required)"""
        def on_key_press(key):
            try:
                if key == pynput_keyboard.Key.esc:
                    self.exit_requested = True
                    self.monitoring = False
                    return False  # Stop listener
            except Exception:
                pass
        
        def on_key_release(key):
            pass
        
        try:
            self._listener = pynput_keyboard.Listener(
                on_press=on_key_press,
                on_release=on_key_release,
                suppress=False  # Don't suppress the key, just detect it
            )
            self._listener.start()
        except Exception:
            # If pynput fails, fall back to simple monitoring
            self._start_simple_monitoring()
    
    def _start_simple_monitoring(self):
        """Simple fallback monitoring using a timeout-based approach"""
        # For this fallback, we'll check periodically and rely on
        # the user understanding they can press Ctrl+C instead
        pass
    
    def is_exit_requested(self) -> bool:
        """Check if exit was requested"""
        return self.exit_requested
    
    def reset(self):
        """Reset the handler"""
        self.exit_requested = False
        self.monitoring = False
        self._stop_event.clear()
        if self._listener:
            try:
                self._listener.stop()
                self._listener = None
            except:
                pass
    
    def request_exit(self):
        """Manually request exit (for fallback methods)"""
        self.exit_requested = True
        self.monitoring = False


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
    
    # RAG-specific methods
    
    async def upload_document(self, file_path: str, file_type: str = None, metadata: dict = None) -> Dict[str, Any]:
        """Upload a document for RAG"""
        from pathlib import Path
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect file type
        if not file_type:
            file_type = file_path.suffix.lower().lstrip('.')
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with latin-1 encoding for some PDFs/text files
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Prepare upload data
        upload_data = {
            "filename": file_path.name,
            "file_type": file_type,
            "content": content,
            "metadata": metadata or {}
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/documents/upload",
            json=upload_data
        )
        response.raise_for_status()
        return response.json()
    
    async def list_documents(self) -> Dict[str, Any]:
        """List uploaded documents"""
        response = await self.client.get(f"{self.base_url}/api/v1/documents")
        response.raise_for_status()
        return response.json()
    
    async def delete_document(self, document_id: str, hard_delete: bool = False) -> Dict[str, Any]:
        """Delete a document"""
        params = {"hard_delete": hard_delete} if hard_delete else {}
        response = await self.client.delete(
            f"{self.base_url}/api/v1/documents/{document_id}",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def get_document_chunks(self, document_id: str) -> Dict[str, Any]:
        """Get document chunks for preview"""
        response = await self.client.get(f"{self.base_url}/api/v1/documents/{document_id}/chunks")
        response.raise_for_status()
        return response.json()
    
    async def search_documents(self, query: str, max_chunks: int = 5, similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """Search documents"""
        search_data = {
            "query": query,
            "k": max_chunks,
            "similarity_threshold": similarity_threshold
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/rag/search",
            json=search_data
        )
        response.raise_for_status()
        return response.json()
    
    async def send_rag_message(
        self,
        message: str,
        model_name: str = None,
        max_tokens: int = 300,
        temperature: float = 0.6,
        use_rag: bool = True,
        max_chunks: int = 5,
        similarity_threshold: float = 0.7,
        document_ids: List[str] = None
    ) -> Dict[str, Any]:
        """Send a RAG-enhanced chat message"""
        data = {
            "message": message,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "use_rag": use_rag,
            "max_chunks": max_chunks,
            "similarity_threshold": similarity_threshold
        }
        
        if self.conversation_id:
            data["conversation_id"] = self.conversation_id
        if model_name:
            data["model_name"] = model_name
        if document_ids:
            data["document_ids"] = document_ids
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/chat/rag",
            json=data
        )
        response.raise_for_status()
        result = response.json()
        
        # Update conversation ID if this was a new conversation
        if not self.conversation_id:
            self.conversation_id = result["conversation_id"]
        
        return result
    
    async def get_rag_info(self) -> Dict[str, Any]:
        """Get RAG configuration and status"""
        response = await self.client.get(f"{self.base_url}/api/v1/chat/rag/info")
        response.raise_for_status()
        return response.json()


class ChatInterface:
    """Rich command-line chat interface with voice input and RAG support"""
    
    def __init__(self, api: ChatAPI):
        self.api = api
        self.history = InMemoryHistory()
        self.current_model = "gpt2"
        self.available_models = {}
        self.chat_models = []
        
        # RAG configuration
        self.rag_enabled = True  # RAG mode toggle
        self.rag_available = False  # Whether RAG is available on server
        self.rag_info = {}
        self.uploaded_documents = []
        self.rag_settings = {
            "max_chunks": 5,
            "similarity_threshold": 0.7,
            "max_tokens": 300
        }
        
        # Voice input support
        self.voice_manager = None
        self.voice_enabled = False
        self.voice_mode = False  # Toggle between text and voice input
        self.voice_exit_handler = VoiceExitHandler()  # Handle Esc key for voice mode
        
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
        """Initialize the chat interface with RAG support"""
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
        
        # Check RAG availability and get info
        try:
            self.rag_info = await self.api.get_rag_info()
            self.rag_available = self.rag_info.get("rag_enabled", False)
            
            if self.rag_available:
                # Get uploaded documents
                docs_data = await self.api.list_documents()
                self.uploaded_documents = docs_data.get("documents", [])
                console.print(f"[green]✅ RAG system available ({len(self.uploaded_documents)} documents loaded)")
            else:
                console.print("[yellow]⚠️ RAG system not available")
                self.rag_enabled = False
                
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not check RAG status: {e}")
            self.rag_available = False
            self.rag_enabled = False
        
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
        """Display welcome message with RAG and voice support"""
        voice_status = ""
        voice_commands = ""
        
        if self.voice_enabled:
            voice_status = "\n[green]🎤 Voice input enabled[/green]"
            esc_note = " (Esc to exit)" if PYNPUT_AVAILABLE else ""
            voice_commands = f"""• [cyan]/voice[/cyan] - Toggle voice input mode{esc_note}
• [cyan]/record[/cyan] - Record a voice message
• [cyan]/voice-settings[/cyan] - Voice configuration
"""
        
        # RAG status and commands
        rag_status = ""
        rag_commands = ""
        
        if self.rag_available:
            rag_mode = "[green]ON[/green]" if self.rag_enabled else "[red]OFF[/red]"
            doc_count = len(self.uploaded_documents)
            rag_status = f"\n[green]📚 RAG system enabled[/green] ({doc_count} documents) - Mode: {rag_mode}"
            rag_commands = f"""• [cyan]/upload <file>[/cyan] - Upload document for RAG
• [cyan]/docs[/cyan] - List uploaded documents
• [cyan]/search <query>[/cyan] - Search documents
• [cyan]/rag on/off[/cyan] - Toggle RAG mode
• [cyan]/rag-info[/cyan] - Show RAG configuration
"""
        
        welcome_text = f"""
[bold blue]🤖 LLM Chat Interface with RAG & Voice Support[/bold blue]
[dim]Connected to: {self.api.base_url}[/dim]{voice_status}{rag_status}

[bold]Available Commands:[/bold]
• [cyan]/help[/cyan] - Show this help
• [cyan]/models[/cyan] - List available models
• [cyan]/switch <model>[/cyan] - Switch models
{rag_commands}{voice_commands}• [cyan]/clear[/cyan] - Clear conversation history  
• [cyan]/save <filename>[/cyan] - Save conversation
• [cyan]/load <filename>[/cyan] - Load conversation
• [cyan]/quit[/cyan] - Exit

[bold]Current Model:[/bold] [green]{self.current_model}[/green]
[bold]Chat Models:[/bold] {', '.join(self.chat_models)}

[dim]Type your message and press Enter to chat!{' RAG will enhance responses with document knowledge.' if self.rag_enabled and self.uploaded_documents else ''}[/dim]
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
        """Send a chat message with optional RAG enhancement"""
        try:
            # Determine if we should use RAG
            use_rag = self.rag_enabled and self.rag_available and len(self.uploaded_documents) > 0
            
            # Show appropriate typing indicator
            rag_indicator = " with document search" if use_rag else ""
            spinner_text = f"[dim]AI is thinking{rag_indicator}...[/dim]"
            
            with Live(Spinner("dots", text=spinner_text), console=console, transient=True):
                if use_rag:
                    # Use RAG-enhanced chat
                    result = await self.api.send_rag_message(
                        message=message,
                        max_tokens=self.rag_settings["max_tokens"],
                        max_chunks=self.rag_settings["max_chunks"],
                        similarity_threshold=self.rag_settings["similarity_threshold"],
                        use_rag=True
                    )
                else:
                    # Use regular chat
                    result = await self.api.send_message(
                        message=message,
                        max_tokens=200
                    )
            
            # Display response with RAG information
            response_text = result["response"]
            model_name = result["model_name"]
            generation_time = result["generation_time"]
            
            # Enhanced subtitle with RAG info
            if use_rag and result.get("rag_used"):
                # RAG was used successfully
                search_results = result.get("search_results", [])
                chunks_found = len(search_results)
                context_length = result.get("context_length", 0)
                
                subtitle = f"🧠 RAG | ⏱️ {generation_time:.2f}s | 📚 {chunks_found} docs | 💬 {result['message_count']} msgs"
                
                # Show search results if available
                if search_results:
                    self._show_rag_sources(search_results)
                    
            elif use_rag and not result.get("rag_used"):
                # RAG was attempted but no relevant documents found
                subtitle = f"⏱️ {generation_time:.2f}s | 💬 {result['message_count']} messages | ⚠️ No relevant docs found"
            else:
                # Regular chat response
                subtitle = f"⏱️ {generation_time:.2f}s | 💬 {result['message_count']} messages"
            
            # Format response panel
            response_panel = Panel(
                response_text,
                title=f"🤖 {model_name}",
                subtitle=subtitle,
                border_style="green" if use_rag and result.get("rag_used") else "blue"
            )
            
            console.print(response_panel)
            
        except Exception as e:
            console.print(f"[red]❌ Chat error: {e}")
    
    def _show_rag_sources(self, search_results: List[Dict]):
        """Display RAG source information"""
        if not search_results:
            return
        
        sources_text = "**Sources used:**\n"
        for i, result in enumerate(search_results[:3], 1):  # Show top 3 sources
            doc_id = result.get("document_id", "Unknown")[:16]
            similarity = result.get("similarity_score", 0)
            page = result.get("page_number")
            
            page_info = f", Page {page}" if page else ""
            sources_text += f"{i}. Doc `{doc_id}`{page_info} (similarity: {similarity:.2f})\n"
        
        console.print(f"[dim]{sources_text.strip()}[/dim]")
    
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
        """Get text input from user with RAG and voice indicators"""
        # Build prompt with status indicators
        prompt_indicators = []
        
        # Add voice mode indicator
        if self.voice_enabled:
            voice_indicator = "🎤" if self.voice_mode else "⌨️"
            prompt_indicators.append(voice_indicator)
        
        # Add RAG status indicator
        if self.rag_available:
            if self.rag_enabled and len(self.uploaded_documents) > 0:
                rag_indicator = f"📚{len(self.uploaded_documents)}"
            elif self.rag_enabled:
                rag_indicator = "📚❌"  # RAG enabled but no docs
            else:
                rag_indicator = "📖"  # RAG disabled
            prompt_indicators.append(rag_indicator)
        
        # Build final prompt
        if prompt_indicators:
            indicators = " ".join(prompt_indicators)
            prompt_text = f"💬 You ({indicators}): "
        else:
            prompt_text = "💬 You: "
        
        return toolkit_prompt(
            prompt_text,
            history=self.history,
            auto_suggest=AutoSuggestFromHistory()
        ).strip()
    
    async def get_voice_input(self) -> Optional[str]:
        """Get voice input from user with real-time feedback, Esc key support, and retry limit"""
        if not self.voice_enabled or not self.voice_manager:
            console.print("[red]❌ Voice input not available")
            return None
        
        max_attempts = 2
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            
            try:
                # Start monitoring for Esc key
                self.voice_exit_handler.start_monitoring()
                
                # Show recording interface with enhanced feedback
                if PYNPUT_AVAILABLE:
                    esc_instruction = "Press [bold red]Esc[/bold red] to exit voice mode"
                else:
                    esc_instruction = "Press [bold red]Ctrl+C[/bold red] to cancel"
                
                # Show attempt number if retrying
                attempt_info = f" (Attempt {attempt}/{max_attempts})" if attempt > 1 else ""
                console.print(f"[bold green]🎤 Recording...{attempt_info} (speak now, will auto-stop on silence)[/bold green]")
                console.print(f"[dim]{esc_instruction}[/dim]")
                
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
                    
                    # Create a custom recording method with progress and exit handling
                    recording_future = asyncio.create_task(
                        self.voice_manager.get_voice_input(mode=VoiceInputMode.AUTO_STOP)
                    )
                    
                    # Update progress while recording and check for exit
                    while not recording_future.done():
                        # Check if user pressed Esc
                        if self.voice_exit_handler.is_exit_requested():
                            console.print("\n[yellow]⚠️ Voice mode exited (Esc pressed)[/yellow]")
                            self.voice_manager.stop_recording()
                            recording_future.cancel()
                            self.voice_mode = False  # Exit voice mode
                            return None
                        
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
                    
                    # Get the result if recording completed normally
                    try:
                        text = await recording_future
                    except asyncio.CancelledError:
                        return None
                
                if text and text.strip():
                    console.print(f"[dim]📝 Transcribed: {text}[/dim]")
                    return text
                else:
                    # No speech detected
                    if attempt < max_attempts:
                        console.print(f"[yellow]⚠️ No speech detected. Retrying... ({attempt}/{max_attempts})[/yellow]")
                        await asyncio.sleep(0.5)  # Brief pause before retry
                    else:
                        console.print(f"[yellow]⚠️ No speech detected after {max_attempts} attempts.[/yellow]")
                        console.print("[dim]💡 Tip: Speak closer to your microphone or check /voice-settings[/dim]")
                        
                        # Optionally exit voice mode after failed attempts
                        self.voice_mode = False
                        console.print("[dim]Returning to text input mode[/dim]")
                        return None
                        
            except KeyboardInterrupt:
                console.print("\n[yellow]Recording cancelled[/yellow]")
                self.voice_manager.stop_recording()
                return None
            except Exception as e:
                console.print(f"[red]❌ Voice input error: {e}")
                return None
            finally:
                # Always stop monitoring when done with this attempt
                self.voice_exit_handler.stop_monitoring()
        
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
        
        if self.voice_mode:
            esc_info = " (Esc to exit voice mode)" if PYNPUT_AVAILABLE else " (Ctrl+C to cancel)"
            console.print(f"[green]✅ Switched to {mode} input mode {icon}{esc_info}")
            console.print("[dim]Speak naturally and the system will auto-detect when you finish[/dim]")
        else:
            console.print(f"[green]✅ Switched to {mode} input mode {icon}")
            # Reset the exit handler when leaving voice mode
            self.voice_exit_handler.reset()
    
    async def record_voice_message(self):
        """Record a single voice message and send it with Esc key support and retry limit"""
        if not self.voice_enabled:
            console.print("[red]❌ Voice input not available")
            return
        
        max_attempts = 2
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            
            try:
                # Start monitoring for Esc key
                self.voice_exit_handler.start_monitoring()
                
                esc_instruction = "Press [bold red]Esc[/bold red] to cancel" if PYNPUT_AVAILABLE else "Press [bold red]Ctrl+C[/bold red] to cancel"
                
                # Show attempt number if retrying
                attempt_info = f" (Attempt {attempt}/{max_attempts})" if attempt > 1 else ""
                console.print(f"[bold green]🎤 Recording voice message...{attempt_info} (speak now, will auto-stop on silence)[/bold green]")
                console.print(f"[dim]{esc_instruction}[/dim]")
                
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
                        # Check if user pressed Esc
                        if self.voice_exit_handler.is_exit_requested():
                            console.print("\n[yellow]⚠️ Recording cancelled (Esc pressed)[/yellow]")
                            self.voice_manager.stop_recording()
                            recording_future.cancel()
                            return
                        
                        elapsed = time.time() - start_time
                        progress.update(recording_task, completed=elapsed)
                        
                        try:
                            level = self.voice_manager.get_current_audio_level()
                            level_bars = self._get_audio_level_bars(level)
                            progress.update(recording_task, description=f"🎤 Voice Message {level_bars}")
                        except:
                            pass
                        
                        await asyncio.sleep(0.1)
                    
                    # Get the result if recording completed normally
                    try:
                        text = await recording_future
                    except asyncio.CancelledError:
                        return
                
                if text and text.strip():
                    console.print(f"[dim]📝 Transcribed: {text}[/dim]")
                    await self.send_message(text)
                    return  # Success, exit the retry loop
                else:
                    # No speech detected
                    if attempt < max_attempts:
                        console.print(f"[yellow]⚠️ No speech detected. Retrying... ({attempt}/{max_attempts})[/yellow]")
                        await asyncio.sleep(0.5)  # Brief pause before retry
                    else:
                        console.print(f"[yellow]⚠️ No speech detected after {max_attempts} attempts.[/yellow]")
                        console.print("[dim]💡 Tip: Speak closer to your microphone or check /voice-settings[/dim]")
                        return
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Recording cancelled[/yellow]")
                self.voice_manager.stop_recording()
                return
            except Exception as e:
                console.print(f"[red]❌ Voice recording error: {e}")
                return
            finally:
                # Always stop monitoring when done with this attempt
                self.voice_exit_handler.stop_monitoring()
    
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
        
        # Voice mode status with Esc key info
        mode_status = "[green]Voice mode[/green]" if self.voice_mode else "[dim]Text mode[/dim]"
        esc_info = " ([red]Esc[/red] to exit)" if self.voice_mode and PYNPUT_AVAILABLE else ""
        
        settings_text = f"""
[bold]🎤 Voice Input Settings[/bold]

[bold]Current Status:[/bold]
• Input mode: {mode_status}{esc_info}
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
• System retries up to 2 times if no speech detected
• All transcription happens locally (offline)
• Use /devices to see available microphones
• Try different Whisper models for better accuracy
{"• Press Esc anytime to exit voice mode gracefully" if PYNPUT_AVAILABLE else "• Press Ctrl+C to cancel recording"}
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
    
    # RAG Implementation Methods
    
    async def upload_document(self, file_path: str):
        """Upload a document for RAG with enhanced feedback"""
        if not self.rag_available:
            console.print("[red]❌ RAG system not available")
            return
        
        try:
            from pathlib import Path
            
            # Validate file exists
            path = Path(file_path)
            if not path.exists():
                console.print(f"[red]❌ File not found: {file_path}")
                return
            
            # Check file size
            file_size = path.stat().st_size
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                console.print(f"[red]❌ File too large: {file_size / 1024 / 1024:.1f}MB (max 50MB)")
                return
            
            # Show upload progress
            with Live(Spinner("dots", text=f"[dim]Uploading {path.name}...[/dim]"), console=console, transient=True):
                result = await self.api.upload_document(file_path)
            
            # Update local document list
            docs_data = await self.api.list_documents()
            self.uploaded_documents = docs_data.get("documents", [])
            
            # Show success with details
            doc_info = result.get("document", {})
            console.print(f"[green]✅ Document uploaded successfully!")
            console.print(f"[dim]ID: {doc_info.get('id', 'Unknown')}")
            console.print(f"[dim]File: {doc_info.get('filename', path.name)}")
            console.print(f"[dim]Type: {doc_info.get('file_type', 'Unknown')}")
            console.print(f"[dim]Size: {file_size / 1024:.1f} KB")
            console.print(f"[dim]Chunks: {doc_info.get('total_chunks', 'Unknown')}")
            
        except FileNotFoundError as e:
            console.print(f"[red]❌ File not found: {e}")
        except Exception as e:
            console.print(f"[red]❌ Upload failed: {e}")
            if "413" in str(e):
                console.print("[yellow]💡 File too large. Try a smaller file.")
            elif "415" in str(e):
                console.print("[yellow]💡 Unsupported file type. Try PDF, TXT, or MD files.")
    
    async def show_documents(self):
        """Display uploaded documents with details"""
        if not self.rag_available:
            console.print("[red]❌ RAG system not available")
            return
        
        try:
            # Refresh document list
            docs_data = await self.api.list_documents()
            self.uploaded_documents = docs_data.get("documents", [])
            
            if not self.uploaded_documents:
                console.print("[yellow]📚 No documents uploaded yet")
                console.print("[dim]Use /upload <file> to add documents for RAG")
                return
            
            # Create documents table
            table = Table(title=f"📚 Uploaded Documents ({len(self.uploaded_documents)})")
            table.add_column("ID", style="cyan", width=12)
            table.add_column("Filename", style="green")
            table.add_column("Type", style="yellow", width=8)
            table.add_column("Size", style="magenta", width=10)
            table.add_column("Chunks", style="blue", width=8)
            table.add_column("Uploaded", style="dim", width=12)
            table.add_column("Status", style="white", width=10)
            
            for doc in self.uploaded_documents:
                # Format upload time
                upload_time = doc.get("created_at", "")
                if upload_time:
                    from datetime import datetime
                    try:
                        dt = datetime.fromisoformat(upload_time.replace('Z', '+00:00'))
                        upload_time = dt.strftime("%m/%d %H:%M")
                    except:
                        pass
                
                # Format file size
                file_size = doc.get("file_size", 0)
                if file_size:
                    if file_size > 1024 * 1024:
                        size_str = f"{file_size / (1024 * 1024):.1f}MB"
                    elif file_size > 1024:
                        size_str = f"{file_size / 1024:.1f}KB"
                    else:
                        size_str = f"{file_size}B"
                else:
                    size_str = "Unknown"
                
                # Status indicator
                status = "✅ Ready" if doc.get("status") == "processed" else "⏳ Processing"
                
                table.add_row(
                    doc.get("id", "")[:12],
                    doc.get("filename", "Unknown"),
                    doc.get("file_type", "").upper(),
                    size_str,
                    str(doc.get("total_chunks", "?")),
                    upload_time,
                    status
                )
            
            console.print(table)
            
            # Show RAG status
            rag_mode = "[green]ON[/green]" if self.rag_enabled else "[red]OFF[/red]"
            console.print(f"\n[dim]RAG Mode: {rag_mode} | Use /rag on|off to toggle[/dim]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to list documents: {e}")
    
    async def search_documents(self, query: str):
        """Search documents and display results"""
        if not self.rag_available:
            console.print("[red]❌ RAG system not available")
            return
        
        if not self.uploaded_documents:
            console.print("[yellow]📚 No documents available for search")
            console.print("[dim]Use /upload <file> to add documents first[/dim]")
            return
        
        try:
            # Show search progress
            with Live(Spinner("dots", text=f"[dim]Searching documents for: '{query}'...[/dim]"), console=console, transient=True):
                result = await self.api.search_documents(
                    query=query,
                    max_chunks=self.rag_settings["max_chunks"],
                    similarity_threshold=self.rag_settings["similarity_threshold"]
                )
            
            # Display search results
            search_results = result.get("results", [])
            search_time = result.get("search_time", 0)
            
            console.print(f"[bold blue]🔍 Search Results for: '{query}'[/bold blue]")
            console.print(f"[dim]Found {len(search_results)} results in {search_time:.3f}s[/dim]\n")
            
            if not search_results:
                console.print("[yellow]📄 No relevant documents found")
                console.print(f"[dim]Try different keywords or lower similarity threshold (current: {self.rag_settings['similarity_threshold']})[/dim]")
                return
            
            # Show each result
            for i, result in enumerate(search_results, 1):
                similarity = result.get("similarity_score", 0)
                content = result.get("content", "")
                doc_id = result.get("document_id", "Unknown")
                page = result.get("page_number")
                
                # Create result panel
                page_info = f" | Page {page}" if page else ""
                subtitle = f"Doc: {doc_id[:16]}...{page_info} | Similarity: {similarity:.3f}"
                
                # Highlight query terms in content (basic)
                display_content = content
                if len(content) > 300:
                    display_content = content[:300] + "..."
                
                result_panel = Panel(
                    display_content,
                    title=f"📄 Result {i}",
                    subtitle=subtitle,
                    border_style="blue" if similarity > 0.8 else "dim"
                )
                console.print(result_panel)
            
        except Exception as e:
            console.print(f"[red]❌ Search failed: {e}")
    
    async def toggle_rag_mode(self, mode: str):
        """Toggle RAG mode on/off"""
        if not self.rag_available:
            console.print("[red]❌ RAG system not available")
            return
        
        mode = mode.lower()
        if mode == "on":
            self.rag_enabled = True
            console.print("[green]✅ RAG mode enabled")
            if not self.uploaded_documents:
                console.print("[yellow]💡 Upload documents with /upload <file> to use RAG")
        elif mode == "off":
            self.rag_enabled = False
            console.print("[yellow]⚠️ RAG mode disabled")
            console.print("[dim]Chat will use standard responses without document context[/dim]")
        else:
            console.print("[red]Usage: /rag <on|off>")
            current_status = "[green]ON[/green]" if self.rag_enabled else "[red]OFF[/red]"
            console.print(f"[yellow]Current status: {current_status}")
    
    async def show_rag_info(self):
        """Display RAG system information and settings"""
        if not self.rag_available:
            console.print("[red]❌ RAG system not available")
            return
        
        try:
            # Get fresh RAG info
            rag_info = await self.api.get_rag_info()
            
            # Document statistics
            doc_count = len(self.uploaded_documents)
            total_chunks = sum(doc.get("total_chunks", 0) for doc in self.uploaded_documents)
            
            # RAG status
            mode_status = "[green]ENABLED[/green]" if self.rag_enabled else "[red]DISABLED[/red]"
            
            # Create info display
            info_text = f"""
[bold]📚 RAG System Information[/bold]

[bold]Status:[/bold]
• RAG Mode: {mode_status}
• System Available: [green]{'Yes' if rag_info.get('rag_enabled') else 'No'}[/green]
• Vector Store: [cyan]{rag_info.get('vector_store_type', 'Unknown')}[/cyan]
• Embedding Model: [cyan]{rag_info.get('embedding_model', 'Unknown')}[/cyan]

[bold]Documents:[/bold]
• Uploaded Documents: [cyan]{doc_count}[/cyan]
• Total Chunks: [cyan]{total_chunks}[/cyan]
• Max File Size: [cyan]{rag_info.get('max_file_size', 'Unknown')}[/cyan]
• Supported Types: [cyan]{', '.join(rag_info.get('supported_types', []))}[/cyan]

[bold]Search Settings:[/bold]
• Max Chunks per Query: [cyan]{self.rag_settings['max_chunks']}[/cyan]
• Similarity Threshold: [cyan]{self.rag_settings['similarity_threshold']}[/cyan]
• Max Response Tokens: [cyan]{self.rag_settings['max_tokens']}[/cyan]

[bold]Performance:[/bold]
• Average Search Time: [cyan]{rag_info.get('avg_search_time', 'N/A')}[/cyan]
• Cache Hit Rate: [cyan]{rag_info.get('cache_hit_rate', 'N/A')}[/cyan]

[bold]Available Commands:[/bold]
• [cyan]/upload <file>[/cyan] - Upload document
• [cyan]/docs[/cyan] - List documents
• [cyan]/search <query>[/cyan] - Search documents
• [cyan]/rag on|off[/cyan] - Toggle RAG mode
• [cyan]/delete-doc <id>[/cyan] - Delete document
            """
            
            console.print(Panel(info_text, title="📚 RAG Information", border_style="blue"))
            
        except Exception as e:
            console.print(f"[red]❌ Failed to get RAG info: {e}")
    
    async def delete_document(self, doc_id: str):
        """Delete a document from RAG system"""
        if not self.rag_available:
            console.print("[red]❌ RAG system not available")
            return
        
        try:
            # Find document by ID (support partial matching)
            matching_docs = [doc for doc in self.uploaded_documents if doc.get("id", "").startswith(doc_id)]
            
            if not matching_docs:
                console.print(f"[red]❌ Document not found: {doc_id}")
                console.print("[yellow]Use /docs to see available document IDs")
                return
            
            if len(matching_docs) > 1:
                console.print(f"[yellow]⚠️ Multiple documents match '{doc_id}':")
                for doc in matching_docs:
                    console.print(f"[cyan]  {doc.get('id', '')[:16]} - {doc.get('filename', 'Unknown')}")
                console.print("[yellow]Please use a more specific ID")
                return
            
            doc = matching_docs[0]
            full_doc_id = doc.get("id", "")
            filename = doc.get("filename", "Unknown")
            
            # Confirm deletion
            if not confirm(f"Delete document '{filename}' ({full_doc_id[:16]}...)?"):
                console.print("[yellow]❌ Deletion cancelled")
                return
            
            # Delete document
            with Live(Spinner("dots", text=f"[dim]Deleting {filename}...[/dim]"), console=console, transient=True):
                result = await self.api.delete_document(full_doc_id)
            
            # Update local document list
            docs_data = await self.api.list_documents()
            self.uploaded_documents = docs_data.get("documents", [])
            
            console.print(f"[green]✅ Document deleted: {filename}")
            console.print(f"[dim]Remaining documents: {len(self.uploaded_documents)}")
            
        except Exception as e:
            console.print(f"[red]❌ Delete failed: {e}")

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
        
        # RAG commands (if available)
        rag_commands = ""
        rag_tips = ""
        
        if self.rag_available:
            rag_status = "[green]ENABLED[/green]" if self.rag_enabled else "[red]DISABLED[/red]"
            doc_count = len(self.uploaded_documents)
            
            rag_commands = f"""
[bold]📚 RAG Commands (Status: {rag_status}, {doc_count} docs):[/bold]
[cyan]/upload <file>[/cyan] - Upload document for RAG (PDF, TXT, MD)
[cyan]/docs[/cyan] - List uploaded documents
[cyan]/search <query>[/cyan] - Search documents  
[cyan]/rag on|off[/cyan] - Toggle RAG mode
[cyan]/rag-info[/cyan] - Show RAG system information
[cyan]/delete-doc <id>[/cyan] - Delete document by ID
"""
            rag_tips = """• Upload documents to enhance chat responses with contextual knowledge
• RAG searches documents automatically during chat when enabled
• Use /search to test document retrieval before chatting
• Larger chunks provide more context but may reduce precision
• """
        
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
            esc_support = "• Press Esc to exit voice mode gracefully" if PYNPUT_AVAILABLE else "• Press Ctrl+C to cancel voice recording"
            voice_tips = f"""• Use /voice to toggle between text and voice input
• Voice input auto-stops on silence detection
• System retries up to 2 times if no speech detected
• All transcription happens locally (offline)
{esc_support}
• """
        
        help_text = f"""
[bold]📚 Available Commands:[/bold]

{base_commands}{rag_commands}{voice_commands}

[bold]💡 Tips:[/bold]
• Use Tab for auto-completion
• Press Ctrl+C to interrupt generation
• Conversations auto-save every 10 messages
• Model switching shows detailed progress for large downloads
• Use Qwen and DeepSeek models for best performance
{rag_tips}{voice_tips}
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
        """Handle user commands including RAG commands"""
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
        
        # RAG commands
        elif cmd == '/upload':
            if len(parts) < 2:
                console.print("[red]Usage: /upload <file_path>")
                console.print("[yellow]Supported formats: PDF, TXT, MD")
            else:
                await self.upload_document(parts[1])
        
        elif cmd == '/docs':
            await self.show_documents()
        
        elif cmd == '/search':
            if len(parts) < 2:
                console.print("[red]Usage: /search <query>")
            else:
                query = " ".join(parts[1:])
                await self.search_documents(query)
        
        elif cmd == '/rag':
            if len(parts) < 2:
                console.print("[red]Usage: /rag <on|off>")
                console.print(f"[yellow]Current status: {'ON' if self.rag_enabled else 'OFF'}")
            else:
                await self.toggle_rag_mode(parts[1])
        
        elif cmd == '/rag-info':
            await self.show_rag_info()
        
        elif cmd == '/delete-doc':
            if len(parts) < 2:
                console.print("[red]Usage: /delete-doc <document_id>")
                console.print("[yellow]Use /docs to see document IDs")
            else:
                await self.delete_document(parts[1])
        
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