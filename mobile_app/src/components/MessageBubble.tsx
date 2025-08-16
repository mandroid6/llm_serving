import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Message } from '../types';
import { chatTheme } from '../utils/theme';

interface MessageBubbleProps {
  message: Message;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.role === 'user';
  
  const containerStyle = [
    styles.container,
    chatTheme.messageContainer,
    isUser ? chatTheme.userMessage : chatTheme.assistantMessage,
  ];

  const textStyle = [
    styles.text,
    { color: isUser ? chatTheme.userMessage.color : chatTheme.assistantMessage.color },
  ];

  const formatTime = (timestamp: Date) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <View style={containerStyle}>
      <Text style={textStyle}>{message.content}</Text>
      <Text style={styles.timestamp}>
        {formatTime(message.timestamp)}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginVertical: 4,
    marginHorizontal: 16,
    padding: 12,
    borderRadius: 16,
    maxWidth: '80%',
  },
  text: {
    fontSize: 16,
    lineHeight: 22,
  },
  timestamp: {
    fontSize: 12,
    color: 'rgba(255,255,255,0.7)',
    marginTop: 4,
    alignSelf: 'flex-end',
  },
});

export default MessageBubble;