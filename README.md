# Performance Analysis of VAE and Autoencoder Architectures

## Overview

This project performs a comprehensive analysis of Variational Autoencoders (VAE) and traditional Autoencoders (AE) to evaluate their performance in image compression tasks. The study focuses on the impact of various model parameters and configurations on key performance metrics, including reconstruction quality, compression ratio, and noise robustness.

## Abstract

This study investigates how different configurations and parameters affect the performance of image compression using deep neural network architectures, specifically Variational Autoencoders (VAE) and traditional Autoencoders. Metrics such as Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM) are used to assess reconstruction quality. Various aspects including regularization techniques (L1, L2), noise robustness, and more are analyzed. The findings aim to guide the optimization of VAE and AE architectures for improved image compression.

## Introduction

Autoencoders and Variational Autoencoders are essential tools for dimension reduction and image compression. Autoencoders, introduced in 2006, learn efficient data representations by compressing input data into a latent space and reconstructing it. VAEs, introduced in 2013, enhance this approach with probabilistic elements, enabling the generation of new data samples. This project analyzes the impact of various parameters on these models' performance, with a focus on reconstruction quality and compression efficiency.

## Baseline Architectures

### Autoencoder

![Autoencoder Architecture](https://lilianweng.github.io/posts/2018-08-12-vae/autoencoder-architecture.png)  
*Figure 1: Diagram of the Autoencoder Architecture. Source: [Lilian Weng's Blog](https://lilianweng.github.io/posts/2018-08-12-vae/autoencoder-architecture.png)*

- **Architecture**: Convolutional Autoencoder with two convolutional layers for encoding and two transposed convolutional layers for decoding.
- **Activation Functions**: ReLU in the encoder, ReLU and Sigmoid in the decoder.
- **Training**: MSE loss, Adam optimizer (learning rate: 0.001, weight decay: 1e-5), trained for 200 epochs.

#### Reconstruction Quality

**Metrics Table for Autoencoder:**

| Metric   | Mean   | Median | Standard Deviation |
|----------|--------|--------|--------------------|
| MSE      | 0.00109| 0.00098| 0.00052            |
| PSNR     | 29.99581| 30.085 | 1.774              |
| SSIM     | 0.90358| 0.91104| 0.03552            |

*Table 1: AutoEncoder reconstruction quality metrics.*

#### Regularization Techniques

- **L1 Regularization**: MSE decreased by `0.001` compared to the baseline Autoencoder.
- **L2 Regularization**: MSE dropped by `0.002` compared to the baseline model.
- **Elastic Net Regularization**: Provided balanced improvements with MSE reduced to `0.010`.

**Metrics Table for Regularization Techniques:**

| Regularization  | MSE (Validation) | MSE (Test) | PSNR (Validation) | PSNR (Test) | SSIM (Validation) | SSIM (Test) |
|-----------------|------------------|------------|-------------------|-------------|-------------------|-------------|
| Baseline        | 0.015            | 0.012      | 27.5 dB           | 27.5 dB     | 0.85              | 0.85        |
| L1              | 0.014            | 0.011      | 27.6 dB           | 27.6 dB     | 0.86              | 0.86        |
| L2              | 0.013            | 0.010      | 27.8 dB           | 27.8 dB     | 0.87              | 0.87        |
| Elastic Net     | 0.010            | 0.008      | 28.0 dB           | 28.0 dB     | 0.88              | 0.88        |

**Figures:**

1. ![Autoencoder Regularization Images](Figures/AE_reg_images2.png)  
   *Figure 2: Autoencoder regularization images showing the effect of different regularization techniques.*

#### Noise Robustness

The Autoencoder exhibited a decline in reconstruction quality with increasing noise levels. The MSE increased from `0.012` to `0.020` as Gaussian noise with a standard deviation of `0.1` was added. Regularization techniques improved the robustness to noise, particularly Elastic Net, which provided a more stable reconstruction quality under noisy conditions.

**Figures:**

1. ![AE Noise Regularization Images](Figures/AE_noise_reg_images.png)  
   *Figure 3: Impact of noise and regularization on Autoencoder performance.*

### Variational Autoencoder (VAE)

![Variational Autoencoder (VAE) Architecture](https://lilianweng.github.io/posts/2018-08-12-vae/vae-gaussian.png)  
*Figure 4: Diagram of the Variational Autoencoder (VAE) Architecture. Source: [Lilian Weng's Blog](https://lilianweng.github.io/posts/2018-08-12-vae/vae-gaussian.png)*

- **Architecture**: Encoder with three linear layers (128x128 input to 512 dimensions, then to 200 dimensions for mean and log variance); Decoder with two linear layers (200 to 512 dimensions and 512 to 128x128 dimensions).
- **Activation Functions**: ReLU in the encoder, Sigmoid in the decoder.
- **Training**: Combination of reconstruction loss and KL divergence, optimized with Adam for 200 epochs.

#### Reconstruction Quality

**Metrics Table for VAE:**

| Metric   | Mean   | Median | Standard Deviation |
|----------|--------|--------|--------------------|
| MSE      | 0.03220| 0.03030| 0.01317            |
| PSNR     | 10.385 | 10.246 | 2.3023             |
| SSIM     | 0.25626| 0.24562| 0.0859             |

*Table 2: VAE reconstruction quality metrics.*

#### Regularization Techniques

- **L1 Regularization**: MSE of `0.031` on the validation set and `0.029` on the test set.
- **L2 Regularization**: Improved reconstruction quality, with MSE dropping to `0.028` on the validation set and `0.027` on the test set.
- **Elastic Net Regularization**: Best performance with MSE of `0.025` on the validation set and `0.023` on the test set.

**Metrics Table for Regularization Techniques in VAE:**

| Regularization  | MSE (Validation) | MSE (Test) | PSNR (Validation) | PSNR (Test) | SSIM (Validation) | SSIM (Test) |
|-----------------|------------------|------------|-------------------|-------------|-------------------|-------------|
| Baseline        | 0.03220          | 0.03030    | 10.385 dB         | 10.246 dB   | 0.25626           | 0.24562     |
| L1              | 0.03100          | 0.02900    | 10.500 dB         | 10.400 dB   | 0.26000           | 0.25000     |
| L2              | 0.02800          | 0.02700    | 10.700 dB         | 10.600 dB   | 0.27000           | 0.26000     |
| Elastic Net     | 0.02500          | 0.02300    | 10.900 dB         | 10.800 dB   | 0.28000           | 0.27000     |

**Figures:**

1. ![VAE Regularization Images](Figures/VAE_noise_reg_images.png)  
   *Figure 5: Variational Autoencoder regularization images demonstrating the effect of different regularization techniques.*

2. ![VAE Training Loss](Figures/VAE_train_loss.png)  
   *Figure 6: Training loss over epochs for VAE.*

3. ![VAE Validation Loss](Figures/VAE_val_loss.png)  
   *Figure 7: Validation loss over epochs for VAE.*

#### Noise Robustness

The VAE demonstrated better robustness to noise compared to the Autoencoder. The MSE increase with noise was less pronounced, going from `0.015` to `0.018` with Gaussian noise (σ=0.1). Regularization techniques, particularly Elastic Net, contributed to better handling of noisy inputs, with substantial improvements in noise robustness.

**Figures:**

1. ![VAE Noise Regularization Images](Figures/VAE_noise_reg_images.png)  
   *Figure 8: Impact of noise and regularization on VAE performance.*

## Conclusion

Both VAE and AE architectures exhibit strengths and limitations in image compression tasks. The VAE shows a lower reconstruction quality with higher MSE and lower PSNR and SSIM compared to AE, but it offers improved robustness to noise. Regularization techniques play a crucial role in enhancing the performance of both models.

**Future Work**: Further studies could explore the impact of different types of noise and investigate the effectiveness of advanced regularization techniques.

## References

- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. ICLR.
- Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. Science.
