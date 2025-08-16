import React, { useEffect } from 'react';
import { 
  View, 
  FlatList, 
  StyleSheet, 
  Alert,
  RefreshControl
} from 'react-native';
import { 
  Text, 
  Card, 
  Button,
  Divider,
  IconButton
} from 'react-native-paper';
import { MaterialIcons } from '@expo/vector-icons';

import { useAppContext } from '../context/AppContext';
import LoadingIndicator from '../components/LoadingIndicator';
import ErrorMessage from '../components/ErrorMessage';
import ModelStatusIndicator from '../components/ModelStatusIndicator';
import { ModelInfo } from '../types';

const ModelsScreen: React.FC = () => {
  const {
    availableModels,
    currentModel,
    isLoading,
    error,
    switchModel,
    loadAvailableModels,
    clearError,
  } = useAppContext();

  const [refreshing, setRefreshing] = React.useState(false);

  useEffect(() => {
    if (availableModels.length === 0) {
      loadAvailableModels();
    }
  }, []);

  const handleModelSwitch = (modelName: string) => {
    if (modelName === currentModel) return;

    Alert.alert(
      'Switch Model',
      `Are you sure you want to switch to ${modelName}? This will affect all new conversations.`,
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Switch', 
          onPress: () => switchModel(modelName),
          style: 'default'
        },
      ]
    );
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await loadAvailableModels();
    } finally {
      setRefreshing(false);
    }
  };

  const getModelIcon = (model: ModelInfo) => {
    if (model.name.includes('gpt2')) return 'psychology';
    if (model.name.includes('llama')) return 'smart-toy';
    if (model.name.includes('qwen')) return 'auto-awesome';
    return 'model-training';
  };

  const formatMemoryRequirement = (memory?: string) => {
    if (!memory) return 'Unknown';
    return memory;
  };

  const formatContextLength = (length?: number) => {
    if (!length) return 'Unknown';
    return length.toLocaleString();
  };

  const renderModelCard = ({ item: model }: { item: ModelInfo }) => {
    const isSelected = model.name === currentModel;
    const isAvailable = model.status === 'available';

    return (
      <Card style={[styles.modelCard, isSelected && styles.selectedCard]}>
        <Card.Content>
          <View style={styles.modelHeader}>
            <View style={styles.modelTitleContainer}>
              <MaterialIcons 
                name={getModelIcon(model) as keyof typeof MaterialIcons.glyphMap} 
                size={24} 
                color={isSelected ? '#1976D2' : '#666'} 
                style={styles.modelIcon}
              />
              <View style={styles.modelInfo}>
                <Text style={[styles.modelName, isSelected && styles.selectedText]}>
                  {model.name}
                </Text>
                {isSelected && (
                  <Text style={styles.currentLabel}>Current Model</Text>
                )}
              </View>
            </View>
            <ModelStatusIndicator 
              model={model} 
              isSelected={isSelected}
            />
          </View>

          {model.description && (
            <Text style={styles.modelDescription}>{model.description}</Text>
          )}

          <View style={styles.modelSpecs}>
            <View style={styles.specItem}>
              <Text style={styles.specLabel}>Memory:</Text>
              <Text style={styles.specValue}>
                {formatMemoryRequirement(model.memory_requirements)}
              </Text>
            </View>
            <View style={styles.specItem}>
              <Text style={styles.specLabel}>Context:</Text>
              <Text style={styles.specValue}>
                {formatContextLength(model.context_length)} tokens
              </Text>
            </View>
          </View>
        </Card.Content>

        <Card.Actions>
          <Button
            mode={isSelected ? "outlined" : "contained"}
            onPress={() => handleModelSwitch(model.name)}
            disabled={!isAvailable || isSelected}
            style={styles.switchButton}
          >
            {isSelected ? 'Selected' : 'Switch to this model'}
          </Button>
        </Card.Actions>
      </Card>
    );
  };

  const renderEmptyState = () => (
    <View style={styles.emptyContainer}>
      <MaterialIcons name="model-training" size={64} color="#ccc" />
      <Text style={styles.emptyTitle}>No Models Available</Text>
      <Text style={styles.emptySubtitle}>
        Unable to load model information. Check your server connection.
      </Text>
      <Button mode="contained" onPress={loadAvailableModels} style={styles.retryButton}>
        Retry
      </Button>
    </View>
  );

  if (isLoading && availableModels.length === 0) {
    return <LoadingIndicator message="Loading available models..." />;
  }

  if (error && availableModels.length === 0) {
    return (
      <ErrorMessage
        error={error}
        onRetry={loadAvailableModels}
        onDismiss={clearError}
      />
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Available Models</Text>
        <Text style={styles.headerSubtitle}>
          Choose an AI model for your conversations
        </Text>
        <IconButton
          icon="refresh"
          size={24}
          onPress={handleRefresh}
          style={styles.refreshButton}
        />
      </View>

      <FlatList
        data={availableModels}
        renderItem={renderModelCard}
        keyExtractor={(item) => item.name}
        contentContainerStyle={styles.modelsList}
        ListEmptyComponent={renderEmptyState}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={handleRefresh}
            colors={['#2196F3']}
          />
        }
        showsVerticalScrollIndicator={false}
      />

      <View style={styles.footer}>
        <Text style={styles.footerText}>
          Current: {currentModel} • {availableModels.length} models available
        </Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  header: {
    backgroundColor: '#fff',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
    position: 'relative',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  refreshButton: {
    position: 'absolute',
    right: 8,
    top: 8,
  },
  modelsList: {
    padding: 16,
    paddingBottom: 80,
  },
  modelCard: {
    marginBottom: 16,
    backgroundColor: '#fff',
    elevation: 2,
  },
  selectedCard: {
    borderWidth: 2,
    borderColor: '#2196F3',
    backgroundColor: '#F3F8FF',
  },
  modelHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  modelTitleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  modelIcon: {
    marginRight: 12,
  },
  modelInfo: {
    flex: 1,
  },
  modelName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  selectedText: {
    color: '#1976D2',
  },
  currentLabel: {
    fontSize: 12,
    color: '#4CAF50',
    fontWeight: '500',
    marginTop: 2,
  },
  modelDescription: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
    marginBottom: 12,
  },
  modelSpecs: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  specItem: {
    flex: 1,
  },
  specLabel: {
    fontSize: 12,
    color: '#999',
    fontWeight: '500',
  },
  specValue: {
    fontSize: 14,
    color: '#333',
    marginTop: 2,
  },
  switchButton: {
    flex: 1,
    marginHorizontal: 8,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginTop: 16,
    marginBottom: 8,
  },
  emptySubtitle: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 24,
  },
  retryButton: {
    minWidth: 120,
  },
  footer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#E0E0E0',
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  footerText: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
});

export default ModelsScreen;