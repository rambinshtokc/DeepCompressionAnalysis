# Performance Analysis of VAE and Autoencoder Architectures

## Overview

This project provides a comprehensive analysis of Variational Autoencoders (VAE) and traditional Autoencoders (AE) for image compression tasks. The analysis focuses on the impact of various model parameters and configurations on key performance metrics, including reconstruction quality, compression ratio, and noise robustness.

## Abstract

This study investigates the effects of different configurations and parameters on the performance of image compression using deep neural network architectures, specifically Variational Autoencoders (VAE) and traditional Autoencoders. Metrics such as Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM) are used to assess reconstruction quality. Various aspects, including regularization techniques (L1, L2), noise robustness, and more, are analyzed to optimize VAE and AE architectures for improved image compression.

## Introduction

Autoencoders and Variational Autoencoders are essential tools for dimension reduction and image compression. Introduced in 2006, Autoencoders learn efficient data representations by compressing input data into a latent space and reconstructing it. VAEs, introduced in 2013, enhance this approach with probabilistic elements, enabling the generation of new data samples. This project analyzes the impact of various parameters on these models' performance, focusing on reconstruction quality and compression efficiency.

## Baseline Architectures

### Autoencoder

![Autoencoder Architecture](https://lilianweng.github.io/posts/2018-08-12-vae/autoencoder-architecture.png)  
*Figure 1: Diagram of the Autoencoder Architecture. Source: [Lilian Weng's Blog](https://lilianweng.github.io/posts/2018-08-12-vae/autoencoder-architecture.png)*

- **Architecture**: Convolutional Autoencoder with two convolutional layers for encoding and two transposed convolutional layers for decoding.
- **Activation Functions**: ReLU in the encoder, ReLU and Sigmoid in the decoder.
- **Training**: MSE loss, Adam optimizer (learning rate: 0.001, weight decay: 1e-5), trained for 200 epochs.

#### Regularization Techniques

- **No Regularization**: MSE = `0.00426`, SSIM = `0.72741`, PSNR = `23.989`
- **L1 Regularization**: MSE = `0.00895`, SSIM = `0.62132`, PSNR = `21.178`
- **L2 Regularization**: MSE = `0.00442`, SSIM = `0.70792`, PSNR = `23.855`
- **L1 + L2 Regularization (Elastic Net)**: MSE = `0.00450`, SSIM = `0.72140`, PSNR = `23.772`

**Metrics Table for Regularization Techniques (Autoencoder):**

| Regularization  | MSE   | SSIM   | PSNR  |
|-----------------|-------|--------|-------|
| No Regularization | 0.00426 | 0.72741 | 23.989 |
| L1               | 0.00895 | 0.62132 | 21.178 |
| L2               | 0.00442 | 0.70792 | 23.855 |
| L1 + L2          | 0.00450 | 0.72140 | 23.772 |

**Figures:**

1. ![Autoencoder Regularization Images](Figures/AE_reg_images2.png)  
   *Figure 2: Autoencoder regularization images showing the effect of different regularization techniques.*

2. ![Autoencoder Training Loss](Figures/AE_loss.png)  
   *Figure 3: Training loss over epochs for Autoencoder.*

#### Noise Robustness

The Autoencoder exhibited a decline in reconstruction quality with increasing noise levels. The MSE increased from `0.012` to `0.020` as Gaussian noise with a standard deviation of `0.1` was added. Regularization techniques improved the robustness to noise, particularly Elastic Net, which provided a more stable reconstruction quality under noisy conditions.

**Figures:**

1. ![AE Noise Regularization Images](Figures/AE_noise_reg_images.png)  
   *Figure 4: Impact of noise and regularization on Autoencoder performance.*

### Variational Autoencoder (VAE)

![Variational Autoencoder (VAE) Architecture](https://lilianweng.github.io/posts/2018-08-12-vae/vae-gaussian.png)  
*Figure 5: Diagram of the Variational Autoencoder (VAE) Architecture. Source: [Lilian Weng's Blog](https://lilianweng.github.io/posts/2018-08-12-vae/vae-gaussian.png)*

- **Architecture**: Encoder with three linear layers (128x128 input to 512 dimensions, then to 200 dimensions for mean and log variance); Decoder with two linear layers (200 to 512 dimensions and 512 to 128x128 dimensions).
- **Activation Functions**: ReLU in the encoder, Sigmoid in the decoder.
- **Training**: Combination of reconstruction loss and KL divergence, optimized with Adam for 200 epochs.

#### Regularization Techniques

**Metrics Table for Regularization Techniques (VAE):**

| Regularization  | MSE    | SSIM   | PSNR  |
|-----------------|--------|--------|-------|
| No Regularization | 0.03220 | 0.25626 | 10.385 |
| L1              | 0.03218 | 0.25657 | 10.374 |
| L2              | 0.04598 | 0.21202 | 7.395  |
| L1 + L2         | 0.06001 | 0.19449 | 5.883  |

*Table 2: VAE reconstruction quality metrics.*

**Figures:**

1. ![VAE Regularization Images](Figures/vae_reg_images_2.png)  
   *Figure 6: Variational Autoencoder regularization images demonstrating the effect of different regularization techniques.*

2. ![VAE Training Loss](Figures/VAE_train_loss.png)  
   *Figure 7: Training loss over epochs for VAE.*

3. ![VAE Validation Loss](Figures/VAE_val_loss.png)  
   *Figure 8: Validation loss over epochs for VAE.*

#### Noise Robustness

The VAE demonstrated better robustness to noise compared to the Autoencoder. The MSE increase with noise was less pronounced, going from `0.015` to `0.018` with Gaussian noise (σ=0.1). Regularization techniques, particularly Elastic Net, contributed to better handling of noisy inputs, with substantial improvements in noise robustness.

**Figures:**

1. ![VAE Noise Regularization Images](Figures/VAE_noise_reg_images.png)  
   *Figure 9: Impact of noise and regularization on VAE performance.*

## Conclusion

Both VAE and AE architectures exhibit strengths and limitations in image compression tasks. The VAE shows a lower reconstruction quality with higher MSE and lower PSNR and SSIM compared to the AE. However, the VAE demonstrates greater robustness to noise, making it a more suitable option for scenarios where noise resilience is critical. Regularization techniques, particularly Elastic Net, play a crucial role in improving both architectures' performance, especially under noisy conditions.

## References

- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.
- Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks.
