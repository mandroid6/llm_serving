import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Text, Chip } from 'react-native-paper';
import { MaterialIcons } from '@expo/vector-icons';
import { ModelInfo } from '../types';

interface ModelStatusIndicatorProps {
  model: ModelInfo;
  isSelected?: boolean;
  onPress?: () => void;
}

const ModelStatusIndicator: React.FC<ModelStatusIndicatorProps> = ({ 
  model, 
  isSelected = false, 
  onPress 
}) => {
  const getStatusColor = () => {
    switch (model.status) {
      case 'available':
        return '#4CAF50';
      case 'loading':
        return '#FF9800';
      case 'error':
        return '#F44336';
      default:
        return '#757575';
    }
  };

  const getStatusIcon = () => {
    switch (model.status) {
      case 'available':
        return 'check-circle';
      case 'loading':
        return 'hourglass-empty';
      case 'error':
        return 'error';
      default:
        return 'help';
    }
  };

  const getStatusText = () => {
    switch (model.status) {
      case 'available':
        return 'Available';
      case 'loading':
        return 'Loading...';
      case 'error':
        return 'Error';
      default:
        return 'Unknown';
    }
  };

  return (
    <Chip
      style={[
        styles.container,
        isSelected && styles.selected,
        { borderColor: getStatusColor() }
      ]}
      textStyle={[
        styles.text,
        isSelected && styles.selectedText
      ]}
      onPress={onPress}
      disabled={model.status !== 'available'}
      icon={() => (
        <MaterialIcons 
          name={getStatusIcon() as keyof typeof MaterialIcons.glyphMap} 
          size={16} 
          color={getStatusColor()} 
        />
      )}
    >
      <View style={styles.content}>
        <Text style={[styles.modelName, isSelected && styles.selectedText]}>
          {model.name}
        </Text>
        <Text style={[styles.status, { color: getStatusColor() }]}>
          {getStatusText()}
        </Text>
      </View>
    </Chip>
  );
};

const styles = StyleSheet.create({
  container: {
    margin: 4,
    borderWidth: 1,
    backgroundColor: '#fff',
  },
  selected: {
    backgroundColor: '#E3F2FD',
    borderWidth: 2,
  },
  content: {
    alignItems: 'center',
  },
  text: {
    color: '#000',
  },
  selectedText: {
    color: '#1976D2',
    fontWeight: 'bold',
  },
  modelName: {
    fontSize: 14,
    fontWeight: '500',
  },
  status: {
    fontSize: 12,
    marginTop: 2,
  },
});

export default ModelStatusIndicator;