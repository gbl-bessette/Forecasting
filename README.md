# Forecasting

Investigation of machine learning based forecasting methods and application of Transformers for time-series prediction. 

Inspired from 2022 paper "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting" by Tian Zhou et. al.
https://doi.org/10.48550/arXiv.2201.12740


Two models have been developed and tested:

1. 
Transformer V1 model uses Discrete Fourier Transforms for Frequency Enhanced Block (FEB-f) and Frequency Enhanced Attention (FEA-f) according to FEDformer. Seasonals and trends have been replaced by shifted data samples and Mixture of Experts blocks have been removed from FEDformer. Transformer V1 model is tested on a sinusoïdal signal containing different frequencies.

2.
Transformer V2 model uses additional U-net structure inside encoder and decoder respectively (similar to the structure of diffusion model architectures). Transformer V2 is trained on TSLA stock market over a short period of time. 


Results: 
- Transformer V1 shows some great results in predicting the shape of sinusoïdal functions, mainly DFT helps in capturing the frequencies of a signal during training.
- Because of the stochastic behaviour of the stock market and, since seasonals and trends were not taken into account, Transformer V2 performs badly on the stock prediction. 


