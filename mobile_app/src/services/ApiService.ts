import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { ChatRequest, ChatResponse, ModelInfo, Conversation } from '../types';

class ApiService {
  private api: AxiosInstance;
  private baseURL: string;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.baseURL = baseURL;
    this.api = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for debugging
    this.api.interceptors.request.use(
      (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor for error handling
    this.api.interceptors.response.use(
      (response) => {
        console.log(`API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('API Response Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // Health check
  async healthCheck(): Promise<boolean> {
    try {
      const response: AxiosResponse = await this.api.get('/api/v1/health');
      return response.status === 200;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  // Chat endpoints
  async sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
    try {
      const response: AxiosResponse<ChatResponse> = await this.api.post('/api/v1/chat', request);
      return response.data;
    } catch (error) {
      console.error('Send chat message failed:', error);
      throw error;
    }
  }

  async createNewConversation(): Promise<{ conversation_id: string }> {
    try {
      const response: AxiosResponse<{ conversation_id: string }> = await this.api.post('/api/v1/chat/new');
      return response.data;
    } catch (error) {
      console.error('Create new conversation failed:', error);
      throw error;
    }
  }

  async getConversation(conversationId: string): Promise<Conversation> {
    try {
      const response: AxiosResponse<Conversation> = await this.api.get(`/api/v1/chat/conversation/${conversationId}`);
      return response.data;
    } catch (error) {
      console.error('Get conversation failed:', error);
      throw error;
    }
  }

  // Model management endpoints
  async getAvailableModels(): Promise<ModelInfo[]> {
    try {
      const response: AxiosResponse<{ models: ModelInfo[] }> = await this.api.get('/api/v1/chat/models');
      return response.data.models;
    } catch (error) {
      console.error('Get available models failed:', error);
      throw error;
    }
  }

  async switchModel(modelName: string): Promise<{ message: string; current_model: string }> {
    try {
      const response: AxiosResponse<{ message: string; current_model: string }> = await this.api.post('/api/v1/chat/switch-model', {
        model_name: modelName,
      });
      return response.data;
    } catch (error) {
      console.error('Switch model failed:', error);
      throw error;
    }
  }

  // Legacy generation endpoint (for compatibility)
  async generateText(prompt: string, options?: any): Promise<{ generated_text: string }> {
    try {
      const response: AxiosResponse<{ generated_text: string }> = await this.api.post('/api/v1/generate', {
        prompt,
        ...options,
      });
      return response.data;
    } catch (error) {
      console.error('Generate text failed:', error);
      throw error;
    }
  }

  // Utility method to update base URL (for connecting to different servers)
  updateBaseURL(newBaseURL: string): void {
    this.baseURL = newBaseURL;
    this.api.defaults.baseURL = newBaseURL;
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default ApiService;