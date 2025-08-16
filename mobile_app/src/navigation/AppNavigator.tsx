import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { MaterialIcons } from '@expo/vector-icons';
import { Platform } from 'react-native';
import { RootStackParamList } from '../types';
import { isWeb, isDesktop, navigationOptions } from '../utils/responsive';

// Import screens
import ChatScreen from '../screens/ChatScreen';
import ModelsScreen from '../screens/ModelsScreen';
import SettingsScreen from '../screens/SettingsScreen';

const Tab = createBottomTabNavigator<RootStackParamList>();

const AppNavigator: React.FC = () => {
  return (
    <Tab.Navigator
      initialRouteName="Chat"
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: keyof typeof MaterialIcons.glyphMap;

          if (route.name === 'Chat') {
            iconName = 'chat';
          } else if (route.name === 'Models') {
            iconName = 'model-training';
          } else if (route.name === 'Settings') {
            iconName = 'settings';
          } else {
            iconName = 'help';
          }

          return <MaterialIcons name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: '#2196F3',
        tabBarInactiveTintColor: 'gray',
        tabBarStyle: {
          paddingBottom: isWeb ? 8 : 5,
          height: isWeb ? (isDesktop() ? 70 : 60) : 60,
          ...(isWeb && {
            borderTopWidth: 1,
            borderTopColor: '#e0e0e0',
            backgroundColor: '#ffffff',
            shadowColor: '#000',
            shadowOffset: { width: 0, height: -2 },
            shadowOpacity: 0.1,
            shadowRadius: 4,
            elevation: 8,
          }),
        },
        tabBarLabelStyle: {
          fontSize: isWeb && isDesktop() ? 14 : 12,
          fontWeight: '500',
        },
        headerStyle: {
          backgroundColor: '#2196F3',
          ...(isWeb && {
            borderBottomWidth: 1,
            borderBottomColor: '#1976D2',
            elevation: 4,
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.1,
            shadowRadius: 4,
          }),
        },
        headerTintColor: '#fff',
        headerTitleStyle: {
          fontWeight: 'bold',
          fontSize: isWeb && isDesktop() ? 20 : 18,
        },
        ...(isWeb && {
          headerTitleAlign: 'center',
        }),
        ...navigationOptions,
      })}
    >
      <Tab.Screen 
        name="Chat" 
        component={ChatScreen}
        options={{
          title: 'Chat',
          headerTitle: 'LLM Chat',
          ...(isWeb && {
            headerTitle: isDesktop() ? 'LLM Chat - AI Assistant' : 'LLM Chat',
          }),
        }}
      />
      <Tab.Screen 
        name="Models" 
        component={ModelsScreen}
        options={{
          title: 'Models',
          headerTitle: isWeb && isDesktop() ? 'AI Model Selection' : 'Model Selection',
        }}
      />
      <Tab.Screen 
        name="Settings" 
        component={SettingsScreen}
        options={{
          title: 'Settings',
          headerTitle: isWeb && isDesktop() ? 'Settings & Conversation History' : 'Settings & History',
        }}
      />
    </Tab.Navigator>
  );
};

export default AppNavigator;