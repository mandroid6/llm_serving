# LLM Chat Mobile App

A React Native mobile interface for the LLM Chat API, providing all functionality available in the CLI interface through a modern mobile app.

## Features

- **Real-time Chat**: Interactive chat interface with the LLM API
- **Model Management**: Switch between available AI models (GPT-2, Llama3, Qwen3)
- **Conversation History**: Save, load, and manage conversation history
- **Offline Storage**: Local storage for conversations and settings
- **Modern UI**: Built with React Native Paper for Material Design
- **Cross-platform**: Works on both iOS and Android

## Prerequisites

- Node.js 18+ and npm/yarn
- Expo CLI (`npm install -g @expo/cli`)
- FastAPI server running on `localhost:8000`
- iOS Simulator (for iOS development) or Android emulator/device

## Installation

1. Navigate to the mobile app directory:
```bash
cd mobile_app
```

2. Install dependencies:
```bash
npm install
```

3. Start the Expo development server:
```bash
npx expo start
```

4. Run on your preferred platform:
   - Press `i` for iOS simulator
   - Press `a` for Android emulator
   - Scan QR code with Expo Go app on your device

## API Connection

The app connects to the FastAPI server at `http://localhost:8000` by default. Make sure your FastAPI server is running:

```bash
# In the root directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Features Overview

### Chat Screen
- Send messages to AI models
- View conversation history
- Real-time typing indicators
- Message bubbles with timestamps
- New conversation button

### Models Screen
- View all available models
- Switch between models
- Model status indicators
- Model specifications (memory, context length)
- Real-time model availability

### Settings Screen
- Server connection status
- Save current conversation
- Load saved conversations
- Conversation history management
- App settings and preferences
- Clear all data option

## CLI Feature Mapping

The mobile app provides all CLI functionality through an intuitive interface:

| CLI Command | Mobile Feature |
|-------------|----------------|
| `/switch <model>` | Models screen with model selection |
| `/save <name>` | Settings screen → Save conversation |
| `/load <name>` | Settings screen → Load conversation |
| `/clear` | Settings screen → Clear current |
| `/models` | Models screen |
| `/help` | Built-in interface help |
| `/quit` | App exit (native) |

## Architecture

- **Frontend**: React Native with Expo
- **Navigation**: React Navigation (Bottom Tabs)
- **State Management**: React Context API
- **UI Framework**: React Native Paper (Material Design)
- **Storage**: AsyncStorage for local persistence
- **API Client**: Axios for HTTP requests

## Project Structure

```
mobile_app/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── MessageBubble.tsx
│   │   ├── MessageInput.tsx
│   │   ├── LoadingIndicator.tsx
│   │   ├── ErrorMessage.tsx
│   │   ├── ModelStatusIndicator.tsx
│   │   └── TypingIndicator.tsx
│   ├── screens/            # Main app screens
│   │   ├── ChatScreen.tsx
│   │   ├── ModelsScreen.tsx
│   │   └── SettingsScreen.tsx
│   ├── services/           # API and external services
│   │   └── ApiService.ts
│   ├── context/            # Global state management
│   │   └── AppContext.tsx
│   ├── utils/              # Utility functions
│   │   ├── StorageService.ts
│   │   └── theme.ts
│   ├── types/              # TypeScript types
│   │   └── index.ts
│   └── navigation/         # Navigation configuration
│       └── AppNavigator.tsx
├── App.tsx                 # Main app component
├── package.json           # Dependencies and scripts
├── app.json              # Expo configuration
├── tsconfig.json         # TypeScript configuration
└── babel.config.js       # Babel configuration
```

## Development

### Adding New Features

1. Create components in `src/components/`
2. Add screens to `src/screens/`
3. Update navigation in `src/navigation/AppNavigator.tsx`
4. Extend context for state management in `src/context/AppContext.tsx`
5. Add API endpoints in `src/services/ApiService.ts`

### Debugging

- Use React Native debugger or browser dev tools
- Enable network inspection for API calls
- Check AsyncStorage in debugger for stored data
- Use Expo logs for runtime errors

### Building for Production

```bash
# Build for production
npx expo build:android
npx expo build:ios

# Or for Expo Application Services (EAS)
eas build --platform android
eas build --platform ios
```

## Troubleshooting

### Common Issues

1. **Server Connection Failed**
   - Ensure FastAPI server is running on localhost:8000
   - Check device/simulator network connectivity
   - Verify API endpoints are accessible

2. **Build Errors**
   - Clear Metro cache: `npx expo start --clear`
   - Delete node_modules and reinstall: `rm -rf node_modules && npm install`
   - Check Expo and React Native versions compatibility

3. **Navigation Issues**
   - Ensure all screen components are properly exported
   - Check navigation types in `src/types/index.ts`

4. **Storage Issues**
   - Clear AsyncStorage: Use Settings → Clear All Data
   - Check storage permissions on device

## Contributing

1. Follow the existing code structure and naming conventions
2. Add TypeScript types for new features
3. Include error handling for all async operations
4. Test on both iOS and Android platforms
5. Update this README for new features

## Dependencies

Key dependencies used in this project:

- **expo**: Cross-platform development framework
- **@react-navigation/native**: Navigation library
- **react-native-paper**: Material Design components
- **@react-native-async-storage/async-storage**: Local storage
- **axios**: HTTP client for API calls
- **@expo/vector-icons**: Icon library

## License

This project is part of the LLM Chat API project and follows the same license.