import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Text, Button, Card } from 'react-native-paper';
import { MaterialIcons } from '@expo/vector-icons';

interface ErrorMessageProps {
  error: string;
  onRetry?: () => void;
  onDismiss?: () => void;
}

const ErrorMessage: React.FC<ErrorMessageProps> = ({ error, onRetry, onDismiss }) => {
  return (
    <Card style={styles.container}>
      <Card.Content>
        <View style={styles.content}>
          <MaterialIcons name="error-outline" size={48} color="#B00020" style={styles.icon} />
          <Text style={styles.title}>Something went wrong</Text>
          <Text style={styles.message}>{error}</Text>
          
          <View style={styles.buttonContainer}>
            {onRetry && (
              <Button 
                mode="contained" 
                onPress={onRetry}
                style={styles.button}
              >
                Retry
              </Button>
            )}
            {onDismiss && (
              <Button 
                mode="outlined" 
                onPress={onDismiss}
                style={styles.button}
              >
                Dismiss
              </Button>
            )}
          </View>
        </View>
      </Card.Content>
    </Card>
  );
};

const styles = StyleSheet.create({
  container: {
    margin: 16,
    backgroundColor: '#fff',
  },
  content: {
    alignItems: 'center',
    padding: 16,
  },
  icon: {
    marginBottom: 16,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#B00020',
    marginBottom: 8,
    textAlign: 'center',
  },
  message: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 16,
    lineHeight: 20,
  },
  buttonContainer: {
    flexDirection: 'row',
    gap: 12,
  },
  button: {
    minWidth: 100,
  },
});

export default ErrorMessage;