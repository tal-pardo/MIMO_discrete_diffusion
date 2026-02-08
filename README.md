MIMO discrete Diffusion: 

System Specifications: 
The Input: A conditional tuple  (xt,σ,Y,H)  where  xt  is the noisy bit sequence,  σ  is the noise level embedding,  Y  is the received complex signal, and  H  is the frame-based Rayleigh channel matrix.
The Output: The model produces soft-bit logits that are converted into final hard bit decisions.

What is Trained: The model parameters are optimized to learn the logits  log p(x0|xt,σ,Y,H) / p(x|xt,σ,Y,H).

The Inference: We use 5-step Iteration to reconstruce  x0, allowing it to "search" for the most likely bit combination given the channel constraints.

Attention & Probability Learning: Self-attention layers compute spatial correlations across antenna streams and temporal dependencies within the 52-symbol OFDM frames.

Channel Progression: The  H  matrix follows a Rayleigh distribution and is re-sampled every 52 symbols.
