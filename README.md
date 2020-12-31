# XBT-Predictors

An implementation of Q-learning applied to (short-term) stock trading. The model uses n-day windows of closing prices to determine if the best action to take at a given time is to buy, sell or sit.

As a result of the short-term state representation, the model is not very good at making decisions over long-term trends, but is quite good at predicting peaks and troughs.

## Motivation

The simple prediction of future prices with RNNs or CNNs is not enough to make mostly correct decisions in the crypto-trading world given the complexity and volatility of such environment. One possible solution could be to use reinforcement learning in combination with some clever deep neural network optimized policies.

This project is an attempt to see if is possible to use reinforcement and Q learning to predict and act super-humanly upon cryptocurrency prices and positions, despite the lack of evidence.




