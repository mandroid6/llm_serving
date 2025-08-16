import React, { useEffect, useRef } from 'react';
import { 
  View, 
  FlatList, 
  StyleSheet, 
  Alert,
  KeyboardAvoidingView,
  Platform
} from 'react-native';
import { 
  Text, 
  FAB, 
  Appbar,
  Snackbar
} from 'react-native-paper';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { useAppContext } from '../context/AppContext';
import MessageBubble from '../components/MessageBubble';
import MessageInput from '../components/MessageInput';
import LoadingIndicator from '../components/LoadingIndicator';
import ErrorMessage from '../components/ErrorMessage';
import TypingIndicator from '../components/TypingIndicator';
import { Message } from '../types';

const ChatScreen: React.FC = () => {
  const {
    currentConversation,
    currentModel,
    isLoading,
    error,
    sendMessage,
    startNewConversation,
    clearError,
    checkServerConnection,
  } = useAppContext();

  const flatListRef = useRef<FlatList>(null);
  const insets = useSafeAreaInsets();
  const [connectionError, setConnectionError] = React.useState(false);

  useEffect(() => {
    checkConnection();
  }, []);

  useEffect(() => {
    // Scroll to bottom when new messages are added
    if (currentConversation?.messages.length) {
      setTimeout(() => {
        flatListRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [currentConversation?.messages]);

  const checkConnection = async () => {
    const isConnected = await checkServerConnection();
    setConnectionError(!isConnected);
  };

  const handleSendMessage = async (message: string) => {
    if (connectionError) {
      Alert.alert(
        'Connection Error',
        'Unable to connect to the server. Please check if the server is running.',
        [
          { text: 'Retry', onPress: checkConnection },
          { text: 'Cancel', style: 'cancel' },
        ]
      );
      return;
    }

    try {
      await sendMessage(message);
    } catch (err) {
      console.error('Send message error:', err);
    }
  };

  const handleNewConversation = () => {
    Alert.alert(
      'New Conversation',
      'Are you sure you want to start a new conversation? Current conversation will be saved.',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Yes', onPress: startNewConversation },
      ]
    );
  };

  const renderMessage = ({ item }: { item: Message }) => (
    <MessageBubble message={item} />
  );

  const renderEmptyState = () => (
    <View style={styles.emptyContainer}>
      <Text style={styles.emptyTitle}>Welcome to LLM Chat!</Text>
      <Text style={styles.emptySubtitle}>
        Start a conversation with the AI assistant
      </Text>
      <Text style={styles.modelInfo}>
        Current model: {currentModel}
      </Text>
    </View>
  );

  if (connectionError) {
    return (
      <View style={styles.container}>
        <ErrorMessage
          error="Unable to connect to the server. Please make sure the FastAPI server is running on localhost:8000."
          onRetry={checkConnection}
        />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <KeyboardAvoidingView 
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.container}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
      >
        {/* Chat Messages */}
        <FlatList
          ref={flatListRef}
          data={currentConversation?.messages || []}
          renderItem={renderMessage}
          keyExtractor={(item) => item.id}
          style={styles.messagesList}
          contentContainerStyle={[
            styles.messagesContainer,
            { paddingBottom: insets.bottom }
          ]}
          ListEmptyComponent={renderEmptyState}
          showsVerticalScrollIndicator={false}
        />

        {/* Typing Indicator */}
        {isLoading && <TypingIndicator />}

        {/* Message Input */}
        <MessageInput
          onSend={handleSendMessage}
          disabled={isLoading}
          placeholder={
            connectionError 
              ? "Server connection required..." 
              : "Type your message..."
          }
        />

        {/* New Conversation FAB */}
        <FAB
          style={[styles.fab, { bottom: insets.bottom + 80 }]}
          icon="plus"
          onPress={handleNewConversation}
          label="New Chat"
        />

        {/* Error Snackbar */}
        <Snackbar
          visible={!!error}
          onDismiss={clearError}
          duration={4000}
          action={{
            label: 'Dismiss',
            onPress: clearError,
          }}
        >
          {error}
        </Snackbar>
      </KeyboardAvoidingView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  messagesList: {
    flex: 1,
  },
  messagesContainer: {
    flexGrow: 1,
    paddingTop: 16,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  emptyTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
    textAlign: 'center',
  },
  emptySubtitle: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 16,
    lineHeight: 24,
  },
  modelInfo: {
    fontSize: 14,
    color: '#2196F3',
    fontWeight: '500',
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
  },
});

export default ChatScreen;