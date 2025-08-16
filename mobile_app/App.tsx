import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { Provider as PaperProvider } from 'react-native-paper';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { NavigationContainer } from '@react-navigation/native';
import { Platform, View, StyleSheet } from 'react-native';

import { AppContextProvider } from './src/context/AppContext';
import AppNavigator from './src/navigation/AppNavigator';
import { theme } from './src/utils/theme';
import { isWeb, webStyles } from './src/utils/responsive';

export default function App() {
  return (
    <SafeAreaProvider>
      <PaperProvider theme={theme}>
        <AppContextProvider>
          <NavigationContainer>
            <View style={[styles.container, isWeb && webStyles.centerContainer]}>
              <AppNavigator />
              <StatusBar style="auto" />
            </View>
          </NavigationContainer>
        </AppContextProvider>
      </PaperProvider>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
    ...(isWeb && {
      minHeight: '100vh',
      width: '100%',
    }),
  },
});