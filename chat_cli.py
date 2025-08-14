#!/usr/bin/env python3
"""
Command-line chat interface for the LLM Serving API with Llama3 support
"""

import asyncio
import json
import sys
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
    """Rich command-line chat interface"""
    
    def __init__(self, api: ChatAPI):
        self.api = api
        self.history = InMemoryHistory()
        self.current_model = "gpt2"
        self.available_models = {}
        self.chat_models = []
        
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
        
        return True
    
    def show_welcome(self):
        """Display welcome message"""
        welcome_text = f"""
[bold blue]🦙 LLM Chat Interface with Llama3[/bold blue]
[dim]Connected to: {self.api.base_url}[/dim]

[bold]Available Commands:[/bold]
• [cyan]/help[/cyan] - Show this help
• [cyan]/models[/cyan] - List available models
• [cyan]/switch <model>[/cyan] - Switch models
• [cyan]/clear[/cyan] - Clear conversation history  
• [cyan]/save <filename>[/cyan] - Save conversation
• [cyan]/load <filename>[/cyan] - Load conversation
• [cyan]/quit[/cyan] - Exit

[bold]Current Model:[/bold] [green]{self.current_model}[/green]
[bold]Chat Models:[/bold] {', '.join(self.chat_models)}

[dim]Type your message and press Enter to chat![/dim]
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
        """Switch to a different model"""
        if model_name not in self.available_models:
            console.print(f"[red]❌ Unknown model: {model_name}")
            console.print(f"[yellow]Available models: {', '.join(self.available_models.keys())}")
            return
        
        if model_name == self.current_model:
            console.print(f"[yellow]Already using model: {model_name}")
            return
        
        try:
            with console.status(f"[bold blue]Switching to {model_name}..."):
                result = await self.api.switch_model(model_name)
            
            self.current_model = model_name
            console.print(f"[green]✅ Switched to {result['model_name']}")
            console.print(f"[dim]Load time: {result.get('load_time', 0):.2f}s[/dim]")
            
            # Start new conversation with new model
            await self.api.new_conversation()
            console.print("[dim]Started new conversation with new model[/dim]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to switch model: {e}")
    
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
    
    def show_help(self):
        """Show help message"""
        help_text = """
[bold]📚 Available Commands:[/bold]

[cyan]/help[/cyan] - Show this help message
[cyan]/models[/cyan] - List all available models
[cyan]/switch <model>[/cyan] - Switch to a different model
[cyan]/clear[/cyan] - Clear conversation history
[cyan]/save [filename][/cyan] - Save conversation (auto-named if no filename)
[cyan]/load <filename>[/cyan] - Load saved conversation
[cyan]/quit[/cyan] or [cyan]/exit[/cyan] - Exit the chat

[bold]💡 Tips:[/bold]
• Use Tab for auto-completion
• Press Ctrl+C to interrupt generation
• Conversations auto-save every 10 messages
• Use Llama3 models for best chat experience

[bold]🦙 Recommended Models:[/bold]
• [green]llama3-1b[/green] - Fast, 4GB RAM required
• [green]llama3-3b[/green] - Higher quality, 8GB RAM required
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
                # Get user input
                user_input = toolkit_prompt(
                    "💬 You: ",
                    history=self.history,
                    auto_suggest=AutoSuggestFromHistory()
                ).strip()
                
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
    """Main entry point that handles event loop detection"""
    try:
        # Try to get the current event loop
        asyncio.get_running_loop()
        # If we get here, there's already a running loop
        console.print("[red]❌ Cannot start chat interface from within an async environment")
        console.print("")
        console.print("[yellow]💡 Solutions:")
        console.print("[cyan]   1. Run from a regular terminal: python chat_cli.py")
        console.print("[cyan]   2. Or use the API directly: curl http://localhost:8000/api/v1/chat/new")
        console.print("[cyan]   3. Or try the web docs: http://localhost:8000/docs")
        console.print("")
        console.print("[dim]Current working directory:", Path.cwd())
        return False
        
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        try:
            asyncio.run(main())
            return True
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
        console.print("[yellow]💡 Try running from a regular terminal:")