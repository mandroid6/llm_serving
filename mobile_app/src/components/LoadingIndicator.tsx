import React from 'react';
import { View, StyleSheet } from 'react-native';
import { ActivityIndicator, Text } from 'react-native-paper';

interface LoadingIndicatorProps {
  message?: string;
  size?: 'small' | 'large';
  color?: string;
}

const LoadingIndicator: React.FC<LoadingIndicatorProps> = ({ 
  message = 'Loading...', 
  size = 'large',
  color = '#2196F3'
}) => {
  return (
    <View style={styles.container}>
      <ActivityIndicator 
        size={size} 
        color={color}
        style={styles.indicator}
      />
      {message && (
        <Text style={styles.message}>{message}</Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  indicator: {
    marginBottom: 16,
  },
  message: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
});

export default LoadingIndicator;