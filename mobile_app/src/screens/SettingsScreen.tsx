import React, { useEffect, useState } from 'react';
import { 
  View, 
  FlatList, 
  StyleSheet, 
  Alert,
  ScrollView
} from 'react-native';
import { 
  Text, 
  Card, 
  Button,
  Divider,
  IconButton,
  Dialog,
  Portal,
  TextInput,
  List,
  Switch
} from 'react-native-paper';
import { MaterialIcons } from '@expo/vector-icons';

import { useAppContext } from '../context/AppContext';
import LoadingIndicator from '../components/LoadingIndicator';
import { Conversation } from '../types';
import { StorageService } from '../utils/StorageService';

const SettingsScreen: React.FC = () => {
  const {
    conversations,
    currentConversation,
    loadConversation,
    deleteConversation,
    saveCurrentConversation,
    loadSavedConversations,
    clearCurrentConversation,
    checkServerConnection,
  } = useAppContext();

  const [saveDialogVisible, setSaveDialogVisible] = useState(false);
  const [conversationName, setConversationName] = useState('');
  const [serverStatus, setServerStatus] = useState<boolean | null>(null);
  const [appSettings, setAppSettings] = useState({
    autoSave: true,
    darkMode: false,
  });

  useEffect(() => {
    loadSavedConversations();
    checkConnection();
    loadAppSettings();
  }, []);

  const loadAppSettings = async () => {
    try {
      const settings = await StorageService.getAppSettings();
      setAppSettings(prev => ({ ...prev, ...settings }));
    } catch (error) {
      console.error('Failed to load app settings:', error);
    }
  };

  const saveAppSettings = async (newSettings: typeof appSettings) => {
    try {
      await StorageService.saveAppSettings(newSettings);
      setAppSettings(newSettings);
    } catch (error) {
      console.error('Failed to save app settings:', error);
    }
  };

  const checkConnection = async () => {
    const isConnected = await checkServerConnection();
    setServerStatus(isConnected);
  };

  const handleSaveCurrentConversation = () => {
    if (!currentConversation || currentConversation.messages.length === 0) {
      Alert.alert('No Conversation', 'There is no active conversation to save.');
      return;
    }
    setSaveDialogVisible(true);
  };

  const confirmSaveConversation = async () => {
    if (!conversationName.trim()) {
      Alert.alert('Name Required', 'Please enter a name for the conversation.');
      return;
    }

    try {
      await saveCurrentConversation(conversationName.trim());
      setSaveDialogVisible(false);
      setConversationName('');
      Alert.alert('Success', 'Conversation saved successfully!');
    } catch (error) {
      Alert.alert('Error', 'Failed to save conversation.');
    }
  };

  const handleLoadConversation = (conversation: Conversation) => {
    Alert.alert(
      'Load Conversation',
      `Load "${conversation.title || 'Untitled'}"? Current conversation will be saved.`,
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Load', 
          onPress: () => loadConversation(conversation.id),
        },
      ]
    );
  };

  const handleDeleteConversation = (conversation: Conversation) => {
    Alert.alert(
      'Delete Conversation',
      `Are you sure you want to delete "${conversation.title || 'Untitled'}"? This action cannot be undone.`,
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Delete', 
          style: 'destructive',
          onPress: () => deleteConversation(conversation.id),
        },
      ]
    );
  };

  const handleClearAllData = () => {
    Alert.alert(
      'Clear All Data',
      'This will delete all saved conversations and settings. This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Clear All', 
          style: 'destructive',
          onPress: async () => {
            try {
              await StorageService.clearAllData();
              clearCurrentConversation();
              await loadSavedConversations();
              Alert.alert('Success', 'All data cleared successfully.');
            } catch (error) {
              Alert.alert('Error', 'Failed to clear data.');
            }
          },
        },
      ]
    );
  };

  const formatConversationDate = (date: Date) => {
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const getConversationPreview = (conversation: Conversation) => {
    const lastMessage = conversation.messages[conversation.messages.length - 1];
    if (!lastMessage) return 'No messages';
    
    const preview = lastMessage.content.substring(0, 100);
    return preview.length < lastMessage.content.length ? preview + '...' : preview;
  };

  const renderConversationItem = ({ item: conversation }: { item: Conversation }) => (
    <Card style={styles.conversationCard}>
      <Card.Content>
        <View style={styles.conversationHeader}>
          <View style={styles.conversationInfo}>
            <Text style={styles.conversationTitle}>
              {conversation.title || 'Untitled Conversation'}
            </Text>
            <Text style={styles.conversationDate}>
              {formatConversationDate(conversation.updated_at)}
            </Text>
            <Text style={styles.conversationPreview}>
              {getConversationPreview(conversation)}
            </Text>
            <Text style={styles.messageCount}>
              {conversation.messages.length} messages
            </Text>
          </View>
          <View style={styles.conversationActions}>
            <IconButton
              icon="folder-open"
              size={20}
              onPress={() => handleLoadConversation(conversation)}
            />
            <IconButton
              icon="delete"
              size={20}
              onPress={() => handleDeleteConversation(conversation)}
            />
          </View>
        </View>
      </Card.Content>
    </Card>
  );

  return (
    <ScrollView style={styles.container}>
      {/* Server Status */}
      <Card style={styles.sectionCard}>
        <Card.Content>
          <List.Item
            title="Server Status"
            description={serverStatus ? "Connected" : "Disconnected"}
            left={() => (
              <MaterialIcons 
                name={serverStatus ? "cloud-done" : "cloud-off"} 
                size={24} 
                color={serverStatus ? "#4CAF50" : "#F44336"} 
              />
            )}
            right={() => (
              <IconButton icon="refresh" onPress={checkConnection} />
            )}
          />
        </Card.Content>
      </Card>

      {/* Current Conversation */}
      <Card style={styles.sectionCard}>
        <Card.Content>
          <Text style={styles.sectionTitle}>Current Conversation</Text>
          {currentConversation ? (
            <View>
              <Text style={styles.currentConvInfo}>
                {currentConversation.messages.length} messages
              </Text>
              <Button 
                mode="contained" 
                onPress={handleSaveCurrentConversation}
                style={styles.saveButton}
              >
                Save Conversation
              </Button>
              <Button 
                mode="outlined" 
                onPress={clearCurrentConversation}
                style={styles.clearButton}
              >
                Clear Current
              </Button>
            </View>
          ) : (
            <Text style={styles.noConversationText}>No active conversation</Text>
          )}
        </Card.Content>
      </Card>

      {/* App Settings */}
      <Card style={styles.sectionCard}>
        <Card.Content>
          <Text style={styles.sectionTitle}>Settings</Text>
          <List.Item
            title="Auto-save Conversations"
            description="Automatically save conversations"
            right={() => (
              <Switch
                value={appSettings.autoSave}
                onValueChange={(value) => 
                  saveAppSettings({ ...appSettings, autoSave: value })
                }
              />
            )}
          />
        </Card.Content>
      </Card>

      {/* Conversation History */}
      <Card style={styles.sectionCard}>
        <Card.Content>
          <View style={styles.historyHeader}>
            <Text style={styles.sectionTitle}>Conversation History</Text>
            <Text style={styles.historyCount}>
              {conversations.length} saved
            </Text>
          </View>
          
          {conversations.length === 0 ? (
            <View style={styles.emptyHistory}>
              <MaterialIcons name="history" size={48} color="#ccc" />
              <Text style={styles.emptyText}>No saved conversations</Text>
            </View>
          ) : (
            <FlatList
              data={conversations}
              renderItem={renderConversationItem}
              keyExtractor={(item) => item.id}
              style={styles.conversationsList}
              scrollEnabled={false}
            />
          )}
        </Card.Content>
      </Card>

      {/* Danger Zone */}
      <Card style={[styles.sectionCard, styles.dangerCard]}>
        <Card.Content>
          <Text style={styles.dangerTitle}>Danger Zone</Text>
          <Button 
            mode="outlined" 
            onPress={handleClearAllData}
            style={styles.dangerButton}
            textColor="#F44336"
          >
            Clear All Data
          </Button>
        </Card.Content>
      </Card>

      {/* Save Dialog */}
      <Portal>
        <Dialog visible={saveDialogVisible} onDismiss={() => setSaveDialogVisible(false)}>
          <Dialog.Title>Save Conversation</Dialog.Title>
          <Dialog.Content>
            <TextInput
              label="Conversation Name"
              value={conversationName}
              onChangeText={setConversationName}
              mode="outlined"
              placeholder="Enter a name for this conversation"
            />
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setSaveDialogVisible(false)}>Cancel</Button>
            <Button onPress={confirmSaveConversation}>Save</Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  sectionCard: {
    margin: 16,
    marginBottom: 8,
    backgroundColor: '#fff',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 16,
  },
  currentConvInfo: {
    fontSize: 14,
    color: '#666',
    marginBottom: 16,
  },
  noConversationText: {
    fontSize: 14,
    color: '#999',
    fontStyle: 'italic',
  },
  saveButton: {
    marginBottom: 8,
  },
  clearButton: {
    borderColor: '#FF9800',
  },
  historyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  historyCount: {
    fontSize: 14,
    color: '#666',
  },
  conversationsList: {
    maxHeight: 400,
  },
  conversationCard: {
    marginBottom: 8,
    backgroundColor: '#FAFAFA',
  },
  conversationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  conversationInfo: {
    flex: 1,
  },
  conversationTitle: {
    fontSize: 16,
    fontWeight: '500',
    color: '#333',
  },
  conversationDate: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  conversationPreview: {
    fontSize: 12,
    color: '#999',
    marginTop: 4,
    lineHeight: 16,
  },
  messageCount: {
    fontSize: 12,
    color: '#2196F3',
    marginTop: 4,
    fontWeight: '500',
  },
  conversationActions: {
    flexDirection: 'row',
  },
  emptyHistory: {
    alignItems: 'center',
    padding: 32,
  },
  emptyText: {
    fontSize: 14,
    color: '#999',
    marginTop: 8,
  },
  dangerCard: {
    borderColor: '#F44336',
    borderWidth: 1,
  },
  dangerTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#F44336',
    marginBottom: 16,
  },
  dangerButton: {
    borderColor: '#F44336',
  },
});

export default SettingsScreen;