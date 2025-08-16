import AsyncStorage from '@react-native-async-storage/async-storage';
import { Conversation } from '../types';

const STORAGE_KEYS = {
  CONVERSATIONS: '@conversations',
  CURRENT_CONVERSATION: '@current_conversation',
  APP_SETTINGS: '@app_settings',
  MODEL_PREFERENCE: '@model_preference',
};

export class StorageService {
  // Conversation management
  static async saveConversation(conversation: Conversation): Promise<void> {
    try {
      const conversations = await this.getAllConversations();
      const existingIndex = conversations.findIndex(c => c.id === conversation.id);
      
      if (existingIndex >= 0) {
        conversations[existingIndex] = conversation;
      } else {
        conversations.push(conversation);
      }
      
      await AsyncStorage.setItem(STORAGE_KEYS.CONVERSATIONS, JSON.stringify(conversations));
    } catch (error) {
      console.error('Failed to save conversation:', error);
      throw error;
    }
  }

  static async getAllConversations(): Promise<Conversation[]> {
    try {
      const conversationsJson = await AsyncStorage.getItem(STORAGE_KEYS.CONVERSATIONS);
      if (conversationsJson) {
        const conversations = JSON.parse(conversationsJson);
        // Convert date strings back to Date objects
        return conversations.map((conv: any) => ({
          ...conv,
          created_at: new Date(conv.created_at),
          updated_at: new Date(conv.updated_at),
          messages: conv.messages.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp),
          })),
        }));
      }
      return [];
    } catch (error) {
      console.error('Failed to get conversations:', error);
      return [];
    }
  }

  static async getConversationById(id: string): Promise<Conversation | null> {
    try {
      const conversations = await this.getAllConversations();
      return conversations.find(c => c.id === id) || null;
    } catch (error) {
      console.error('Failed to get conversation by ID:', error);
      return null;
    }
  }

  static async deleteConversation(id: string): Promise<void> {
    try {
      const conversations = await this.getAllConversations();
      const filteredConversations = conversations.filter(c => c.id !== id);
      await AsyncStorage.setItem(STORAGE_KEYS.CONVERSATIONS, JSON.stringify(filteredConversations));
    } catch (error) {
      console.error('Failed to delete conversation:', error);
      throw error;
    }
  }

  static async saveConversationWithName(conversation: Conversation, name: string): Promise<void> {
    try {
      const conversationWithTitle = {
        ...conversation,
        title: name,
        updated_at: new Date(),
      };
      await this.saveConversation(conversationWithTitle);
    } catch (error) {
      console.error('Failed to save conversation with name:', error);
      throw error;
    }
  }

  // Current conversation state
  static async setCurrentConversation(conversation: Conversation | null): Promise<void> {
    try {
      if (conversation) {
        await AsyncStorage.setItem(STORAGE_KEYS.CURRENT_CONVERSATION, JSON.stringify(conversation));
      } else {
        await AsyncStorage.removeItem(STORAGE_KEYS.CURRENT_CONVERSATION);
      }
    } catch (error) {
      console.error('Failed to set current conversation:', error);
      throw error;
    }
  }

  static async getCurrentConversation(): Promise<Conversation | null> {
    try {
      const conversationJson = await AsyncStorage.getItem(STORAGE_KEYS.CURRENT_CONVERSATION);
      if (conversationJson) {
        const conversation = JSON.parse(conversationJson);
        // Convert date strings back to Date objects
        return {
          ...conversation,
          created_at: new Date(conversation.created_at),
          updated_at: new Date(conversation.updated_at),
          messages: conversation.messages.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp),
          })),
        };
      }
      return null;
    } catch (error) {
      console.error('Failed to get current conversation:', error);
      return null;
    }
  }

  // App settings
  static async saveAppSettings(settings: any): Promise<void> {
    try {
      await AsyncStorage.setItem(STORAGE_KEYS.APP_SETTINGS, JSON.stringify(settings));
    } catch (error) {
      console.error('Failed to save app settings:', error);
      throw error;
    }
  }

  static async getAppSettings(): Promise<any> {
    try {
      const settingsJson = await AsyncStorage.getItem(STORAGE_KEYS.APP_SETTINGS);
      return settingsJson ? JSON.parse(settingsJson) : {};
    } catch (error) {
      console.error('Failed to get app settings:', error);
      return {};
    }
  }

  // Model preference
  static async saveModelPreference(modelName: string): Promise<void> {
    try {
      await AsyncStorage.setItem(STORAGE_KEYS.MODEL_PREFERENCE, modelName);
    } catch (error) {
      console.error('Failed to save model preference:', error);
      throw error;
    }
  }

  static async getModelPreference(): Promise<string | null> {
    try {
      return await AsyncStorage.getItem(STORAGE_KEYS.MODEL_PREFERENCE);
    } catch (error) {
      console.error('Failed to get model preference:', error);
      return null;
    }
  }

  // Utility methods
  static async clearAllData(): Promise<void> {
    try {
      await AsyncStorage.multiRemove(Object.values(STORAGE_KEYS));
    } catch (error) {
      console.error('Failed to clear all data:', error);
      throw error;
    }
  }

  static async getStorageInfo(): Promise<{ [key: string]: string | null }> {
    try {
      const keys = Object.values(STORAGE_KEYS);
      const values = await AsyncStorage.multiGet(keys);
      return Object.fromEntries(values);
    } catch (error) {
      console.error('Failed to get storage info:', error);
      return {};
    }
  }
}