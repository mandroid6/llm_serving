export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

export interface Conversation {
  id: string;
  messages: Message[];
  created_at: Date;
  updated_at: Date;
  title?: string;
}

export interface ModelInfo {
  name: string;
  description: string;
  status: 'available' | 'loading' | 'error';
  memory_requirements?: string;
  context_length?: number;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  system_prompt?: string;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  message_id: string;
}

export interface AppState {
  currentConversation: Conversation | null;
  conversations: Conversation[];
  availableModels: ModelInfo[];
  currentModel: string;
  isLoading: boolean;
  error: string | null;
}

export type RootStackParamList = {
  Chat: undefined;
  Models: undefined;
  Settings: undefined;
  ConversationHistory: undefined;
};