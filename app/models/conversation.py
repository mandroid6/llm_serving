"""
Conversation management for chat interface
"""
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path


class Message:
    """Represents a single message in a conversation"""
    
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None):
        self.role = role  # 'system', 'user', 'assistant'
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary"""
        msg = cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
        msg.id = data.get("id", str(uuid.uuid4()))
        return msg


class Conversation:
    """Manages a conversation with message history"""
    
    def __init__(self, system_prompt: Optional[str] = None, max_length: int = 50):
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.messages: List[Message] = []
        self.max_length = max_length
        self.title = "New Conversation"
        
        # Add system message if provided
        if system_prompt:
            self.add_system_message(system_prompt)
    
    def add_system_message(self, content: str) -> None:
        """Add or update system message"""
        # Remove existing system message if any
        self.messages = [msg for msg in self.messages if msg.role != "system"]
        # Add new system message at the beginning
        self.messages.insert(0, Message("system", content))
        self._update_timestamp()
    
    def add_user_message(self, content: str) -> None:
        """Add user message to conversation"""
        self.messages.append(Message("user", content))
        self._update_timestamp()
        self._trim_conversation()
        
        # Set title from first user message if not set
        if self.title == "New Conversation" and content:
            self.title = content[:50] + ("..." if len(content) > 50 else "")
    
    def add_assistant_message(self, content: str) -> None:
        """Add assistant message to conversation"""
        self.messages.append(Message("assistant", content))
        self._update_timestamp()
        self._trim_conversation()
    
    def get_messages_for_model(self) -> List[Dict[str, str]]:
        """Get messages in format expected by model"""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]
    
    def get_context_messages(self, include_system: bool = True) -> List[Dict[str, str]]:
        """Get messages for context, optionally excluding system message"""
        messages = self.get_messages_for_model()
        if not include_system:
            messages = [msg for msg in messages if msg["role"] != "system"]
        return messages
    
    def clear_history(self, keep_system: bool = True) -> None:
        """Clear conversation history"""
        if keep_system:
            system_messages = [msg for msg in self.messages if msg.role == "system"]
            self.messages = system_messages
        else:
            self.messages = []
        self._update_timestamp()
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message"""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None
    
    def get_message_count(self) -> Dict[str, int]:
        """Get count of messages by role"""
        counts = {"system": 0, "user": 0, "assistant": 0}
        for msg in self.messages:
            counts[msg.role] = counts.get(msg.role, 0) + 1
        return counts
    
    def _trim_conversation(self) -> None:
        """Trim conversation to max length, keeping system message"""
        if len(self.messages) <= self.max_length:
            return
        
        # Find system message
        system_msg = None
        other_messages = []
        
        for msg in self.messages:
            if msg.role == "system":
                system_msg = msg
            else:
                other_messages.append(msg)
        
        # Keep only the most recent messages
        if len(other_messages) > self.max_length - (1 if system_msg else 0):
            other_messages = other_messages[-(self.max_length - (1 if system_msg else 0)):]
        
        # Reconstruct messages list
        self.messages = []
        if system_msg:
            self.messages.append(system_msg)
        self.messages.extend(other_messages)
    
    def _update_timestamp(self) -> None:
        """Update the conversation timestamp"""
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export conversation to dictionary"""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "max_length": self.max_length,
            "messages": [msg.to_dict() for msg in self.messages],
            "message_count": self.get_message_count()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """Load conversation from dictionary"""
        conv = cls(max_length=data.get("max_length", 50))
        conv.id = data["id"]
        conv.title = data.get("title", "Loaded Conversation")
        conv.created_at = datetime.fromisoformat(data["created_at"])
        conv.updated_at = datetime.fromisoformat(data["updated_at"])
        conv.messages = [Message.from_dict(msg_data) for msg_data in data["messages"]]
        return conv
    
    def save_to_file(self, filepath: Path) -> None:
        """Save conversation to JSON file"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> "Conversation":
        """Load conversation from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)