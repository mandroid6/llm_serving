import { Dimensions, Platform } from 'react-native';

export const isWeb = Platform.OS === 'web';
export const isNative = Platform.OS !== 'web';

const { width, height } = Dimensions.get('window');

export const screenWidth = width;
export const screenHeight = height;

// Responsive breakpoints for web
export const breakpoints = {
  mobile: 480,
  tablet: 768,
  desktop: 1024,
  largeDesktop: 1440,
};

export const getScreenType = () => {
  if (width >= breakpoints.largeDesktop) return 'largeDesktop';
  if (width >= breakpoints.desktop) return 'desktop';
  if (width >= breakpoints.tablet) return 'tablet';
  return 'mobile';
};

export const isDesktop = () => {
  return width >= breakpoints.desktop;
};

export const isTablet = () => {
  return width >= breakpoints.tablet && width < breakpoints.desktop;
};

export const isMobile = () => {
  return width < breakpoints.tablet;
};

// Web-specific styles
export const webStyles = {
  maxWidth: isWeb ? 1200 : '100%',
  centerContainer: isWeb ? {
    marginHorizontal: 'auto',
    maxWidth: 1200,
  } : {},
  
  // Chat specific responsive styles
  chatContainer: isWeb ? {
    flexDirection: 'row' as const,
    maxWidth: 1200,
    marginHorizontal: 'auto',
  } : {
    flex: 1,
  },
  
  chatMain: isWeb && isDesktop() ? {
    flex: 1,
    maxWidth: 800,
  } : {
    flex: 1,
  },
  
  chatSidebar: isWeb && isDesktop() ? {
    width: 300,
    borderLeftWidth: 1,
    borderLeftColor: '#e0e0e0',
  } : null,
  
  // Card responsive styles
  cardContainer: isWeb ? {
    maxWidth: isDesktop() ? 600 : '100%',
    alignSelf: 'center' as const,
  } : {},
  
  // Input responsive styles  
  messageInput: isWeb ? {
    maxWidth: 800,
    alignSelf: 'center' as const,
  } : {},
};

// Platform-specific navigation behavior
export const navigationOptions = {
  // On web, we might want different navigation behavior
  gestureEnabled: isNative,
  animationEnabled: isNative,
  
  // Web-specific header options
  headerStyle: isWeb ? {
    backgroundColor: '#2196F3',
    borderBottomWidth: 1,
    borderBottomColor: '#1976D2',
  } : {
    backgroundColor: '#2196F3',
  },
};

// Platform utilities
export const getPlatformValue = <T>(values: {
  web?: T;
  native?: T;
  ios?: T;
  android?: T;
  default: T;
}): T => {
  if (isWeb && values.web !== undefined) return values.web;
  if (Platform.OS === 'ios' && values.ios !== undefined) return values.ios;
  if (Platform.OS === 'android' && values.android !== undefined) return values.android;
  if (isNative && values.native !== undefined) return values.native;
  return values.default;
};