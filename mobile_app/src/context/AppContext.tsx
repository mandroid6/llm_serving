import React, { createContext, useContext, useReducer, useEffect, ReactNode } from 'react';
import { AppState, Conversation, Message, ModelInfo } from '../types';
import { apiService } from '../services/ApiService';
import { StorageService } from '../utils/StorageService';

// Action types
type AppAction =
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_CURRENT_CONVERSATION'; payload: Conversation | null }
  | { type: 'ADD_MESSAGE'; payload: Message }
  | { type: 'SET_CONVERSATIONS'; payload: Conversation[] }
  | { type: 'SET_AVAILABLE_MODELS'; payload: ModelInfo[] }
  | { type: 'SET_CURRENT_MODEL'; payload: string }
  | { type: 'CLEAR_CURRENT_CONVERSATION' }
  | { type: 'UPDATE_CONVERSATION'; payload: Conversation };

// Initial state
const initialState: AppState = {
  currentConversation: null,
  conversations: [],
  availableModels: [],
  currentModel: 'qwen3-1.8b',
  isLoading: false,
  error: null,
};

// Reducer
const appReducer = (state: AppState, action: AppAction): AppState => {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    
    case 'SET_ERROR':
      return { ...state, error: action.payload, isLoading: false };
    
    case 'SET_CURRENT_CONVERSATION':
      return { ...state, currentConversation: action.payload };
    
    case 'ADD_MESSAGE':
      if (!state.currentConversation) return state;
      const updatedConversation = {
        ...state.currentConversation,
        messages: [...state.currentConversation.messages, action.payload],
        updated_at: new Date(),
      };
      return { ...state, currentConversation: updatedConversation };
    
    case 'SET_CONVERSATIONS':
      return { ...state, conversations: action.payload };
    
    case 'SET_AVAILABLE_MODELS':
      return { ...state, availableModels: action.payload };
    
    case 'SET_CURRENT_MODEL':
      return { ...state, currentModel: action.payload };
    
    case 'CLEAR_CURRENT_CONVERSATION':
      return { ...state, currentConversation: null };
    
    case 'UPDATE_CONVERSATION':
      const updatedConversations = state.conversations.map(conv =>
        conv.id === action.payload.id ? action.payload : conv
      );
      return {
        ...state,
        conversations: updatedConversations,
        currentConversation: state.currentConversation?.id === action.payload.id 
          ? action.payload 
          : state.currentConversation,
      };
    
    default:
      return state;
  }
};

// Context and Actions interface
interface AppContextType extends AppState {
  // Chat actions
  sendMessage: (message: string) => Promise<void>;
  startNewConversation: () => Promise<void>;
  loadConversation: (conversationId: string) => Promise<void>;
  clearCurrentConversation: () => void;
  
  // Model actions
  switchModel: (modelName: string) => Promise<void>;
  loadAvailableModels: () => Promise<void>;
  
  // Storage actions
  saveCurrentConversation: (name?: string) => Promise<void>;
  loadSavedConversations: () => Promise<void>;
  deleteConversation: (conversationId: string) => Promise<void>;
  
  // Utility actions
  clearError: () => void;
  checkServerConnection: () => Promise<boolean>;
}

// Create context
const AppContext = createContext<AppContextType | undefined>(undefined);

// Provider component
interface AppContextProviderProps {
  children: ReactNode;
}

export const AppContextProvider: React.FC<AppContextProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Initialize app
  useEffect(() => {
    initializeApp();
  }, []);

  const initializeApp = async () => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      
      // Load saved conversations
      await loadSavedConversations();
      
      // Load current conversation
      const currentConv = await StorageService.getCurrentConversation();
      if (currentConv) {
        dispatch({ type: 'SET_CURRENT_CONVERSATION', payload: currentConv });
      }
      
      // Load model preference
      const modelPreference = await StorageService.getModelPreference();
      if (modelPreference) {
        dispatch({ type: 'SET_CURRENT_MODEL', payload: modelPreference });
      }
      
      // Load available models
      await loadAvailableModels();
      
    } catch (error) {
      console.error('Failed to initialize app:', error);
      dispatch({ type: 'SET_ERROR', payload: 'Failed to initialize app' });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  // Chat actions
  const sendMessage = async (message: string) => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'SET_ERROR', payload: null });

      // Create user message
      const userMessage: Message = {
        id: Date.now().toString(),
        content: message,
        role: 'user',
        timestamp: new Date(),
      };

      dispatch({ type: 'ADD_MESSAGE', payload: userMessage });

      // If no current conversation, create one
      let conversationId = state.currentConversation?.id;
      if (!conversationId) {
        const newConv = await apiService.createNewConversation();
        conversationId = newConv.conversation_id;
        
        const conversation: Conversation = {
          id: conversationId,
          messages: [userMessage],
          created_at: new Date(),
          updated_at: new Date(),
        };
        
        dispatch({ type: 'SET_CURRENT_CONVERSATION', payload: conversation });
      }

      // Send message to API
      const response = await apiService.sendChatMessage({
        message,
        conversation_id: conversationId,
      });

      // Add assistant message
      const assistantMessage: Message = {
        id: response.message_id,
        content: response.response,
        role: 'assistant',
        timestamp: new Date(),
      };

      dispatch({ type: 'ADD_MESSAGE', payload: assistantMessage });

      // Auto-save conversation
      if (state.currentConversation) {
        await StorageService.setCurrentConversation(state.currentConversation);
      }

    } catch (error) {
      console.error('Failed to send message:', error);
      dispatch({ type: 'SET_ERROR', payload: 'Failed to send message' });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  const startNewConversation = async () => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      
      // Save current conversation if it exists
      if (state.currentConversation) {
        await StorageService.saveConversation(state.currentConversation);
      }
      
      // Clear current conversation
      dispatch({ type: 'CLEAR_CURRENT_CONVERSATION' });
      await StorageService.setCurrentConversation(null);
      
    } catch (error) {
      console.error('Failed to start new conversation:', error);
      dispatch({ type: 'SET_ERROR', payload: 'Failed to start new conversation' });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  const loadConversation = async (conversationId: string) => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      
      const conversation = await StorageService.getConversationById(conversationId);
      if (conversation) {
        dispatch({ type: 'SET_CURRENT_CONVERSATION', payload: conversation });
        await StorageService.setCurrentConversation(conversation);
      }
      
    } catch (error) {
      console.error('Failed to load conversation:', error);
      dispatch({ type: 'SET_ERROR', payload: 'Failed to load conversation' });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  const clearCurrentConversation = () => {
    dispatch({ type: 'CLEAR_CURRENT_CONVERSATION' });
    StorageService.setCurrentConversation(null);
  };

  // Model actions
  const switchModel = async (modelName: string) => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      
      await apiService.switchModel(modelName);
      dispatch({ type: 'SET_CURRENT_MODEL', payload: modelName });
      await StorageService.saveModelPreference(modelName);
      
    } catch (error) {
      console.error('Failed to switch model:', error);
      dispatch({ type: 'SET_ERROR', payload: 'Failed to switch model' });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  const loadAvailableModels = async () => {
    try {
      const models = await apiService.getAvailableModels();
      dispatch({ type: 'SET_AVAILABLE_MODELS', payload: models });
    } catch (error) {
      console.error('Failed to load available models:', error);
      dispatch({ type: 'SET_ERROR', payload: 'Failed to load available models' });
    }
  };

  // Storage actions
  const saveCurrentConversation = async (name?: string) => {
    try {
      if (!state.currentConversation) return;
      
      if (name) {
        await StorageService.saveConversationWithName(state.currentConversation, name);
      } else {
        await StorageService.saveConversation(state.currentConversation);
      }
      
      await loadSavedConversations();
      
    } catch (error) {
      console.error('Failed to save conversation:', error);
      dispatch({ type: 'SET_ERROR', payload: 'Failed to save conversation' });
    }
  };

  const loadSavedConversations = async () => {
    try {
      const conversations = await StorageService.getAllConversations();
      dispatch({ type: 'SET_CONVERSATIONS', payload: conversations });
    } catch (error) {
      console.error('Failed to load saved conversations:', error);
      dispatch({ type: 'SET_ERROR', payload: 'Failed to load saved conversations' });
    }
  };

  const deleteConversation = async (conversationId: string) => {
    try {
      await StorageService.deleteConversation(conversationId);
      await loadSavedConversations();
      
      // Clear current conversation if it was deleted
      if (state.currentConversation?.id === conversationId) {
        dispatch({ type: 'CLEAR_CURRENT_CONVERSATION' });
        await StorageService.setCurrentConversation(null);
      }
      
    } catch (error) {
      console.error('Failed to delete conversation:', error);
      dispatch({ type: 'SET_ERROR', payload: 'Failed to delete conversation' });
    }
  };

  // Utility actions
  const clearError = () => {
    dispatch({ type: 'SET_ERROR', payload: null });
  };

  const checkServerConnection = async (): Promise<boolean> => {
    try {
      return await apiService.healthCheck();
    } catch (error) {
      return false;
    }
  };

  const contextValue: AppContextType = {
    ...state,
    sendMessage,
    startNewConversation,
    loadConversation,
    clearCurrentConversation,
    switchModel,
    loadAvailableModels,
    saveCurrentConversation,
    loadSavedConversations,
    deleteConversation,
    clearError,
    checkServerConnection,
  };

  return (
    <AppContext.Provider value={contextValue}>
      {children}
    </AppContext.Provider>
  );
};

// Hook to use the context
export const useAppContext = (): AppContextType => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppContext must be used within an AppContextProvider');
  }
  return context;
};