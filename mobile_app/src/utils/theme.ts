import { MD3LightTheme as DefaultTheme } from 'react-native-paper';

export const theme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    primary: '#2196F3',
    secondary: '#03DAC6',
    surface: '#FFFFFF',
    background: '#F5F5F5',
    error: '#B00020',
    text: '#000000',
    onSurface: '#000000',
    disabled: '#BDBDBD',
    placeholder: '#757575',
  },
};

export const chatTheme = {
  userMessage: {
    backgroundColor: '#2196F3',
    color: '#FFFFFF',
    alignSelf: 'flex-end' as const,
  },
  assistantMessage: {
    backgroundColor: '#E0E0E0',
    color: '#000000',
    alignSelf: 'flex-start' as const,
  },
  messageContainer: {
    marginVertical: 4,
    marginHorizontal: 16,
    padding: 12,
    borderRadius: 16,
    maxWidth: '80%',
  },
};