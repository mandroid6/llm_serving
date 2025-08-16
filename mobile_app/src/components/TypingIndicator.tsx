import React, { useEffect, useRef } from 'react';
import { View, StyleSheet, Animated } from 'react-native';
import { Text } from 'react-native-paper';

const TypingIndicator: React.FC = () => {
  const dot1 = useRef(new Animated.Value(0)).current;
  const dot2 = useRef(new Animated.Value(0)).current;
  const dot3 = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const animateDot = (dot: Animated.Value, delay: number) => {
      return Animated.loop(
        Animated.sequence([
          Animated.delay(delay),
          Animated.timing(dot, {
            toValue: 1,
            duration: 600,
            useNativeDriver: true,
          }),
          Animated.timing(dot, {
            toValue: 0,
            duration: 600,
            useNativeDriver: true,
          }),
        ])
      );
    };

    const animation = Animated.parallel([
      animateDot(dot1, 0),
      animateDot(dot2, 200),
      animateDot(dot3, 400),
    ]);

    animation.start();

    return () => animation.stop();
  }, []);

  const animatedStyle = (dot: Animated.Value) => ({
    opacity: dot,
    transform: [
      {
        scale: dot.interpolate({
          inputRange: [0, 1],
          outputRange: [1, 1.2],
        }),
      },
    ],
  });

  return (
    <View style={styles.container}>
      <Text style={styles.text}>Assistant is typing</Text>
      <View style={styles.dotsContainer}>
        <Animated.View style={[styles.dot, animatedStyle(dot1)]} />
        <Animated.View style={[styles.dot, animatedStyle(dot2)]} />
        <Animated.View style={[styles.dot, animatedStyle(dot3)]} />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    alignSelf: 'flex-start',
    marginLeft: 16,
    marginVertical: 4,
    backgroundColor: '#E0E0E0',
    borderRadius: 16,
    maxWidth: '80%',
  },
  text: {
    fontSize: 14,
    color: '#666',
    marginRight: 8,
    fontStyle: 'italic',
  },
  dotsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#666',
    marginHorizontal: 1,
  },
});

export default TypingIndicator;