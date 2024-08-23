# Performance Analysis of VAE and Autoencoder Architectures

## Overview

This project provides a comprehensive analysis of Variational Autoencoders (VAE) and traditional Autoencoders (AE) for image compression tasks. The analysis focuses on the impact of various model parameters and configurations on key performance metrics, including reconstruction quality, compression ratio, and noise robustness.

## Abstract

This study investigates the effects of different configurations and parameters on the performance of image compression using deep neural network architectures, specifically Variational Autoencoders (VAE) and traditional Autoencoders. Metrics such as Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM) are used to assess reconstruction quality. Various aspects, including regularization techniques (L1, L2), noise robustness, and more, are analyzed to optimize VAE and AE architectures for improved image compression.

## Introduction

Autoencoders and Variational Autoencoders are essential tools for dimension reduction and image compression. Introduced in 2006, Autoencoders learn efficient data representations by compressing input data into a latent space and reconstructing it. VAEs, introduced in 2013 [Kingma & Welling](https://arxiv.org/abs/1312.6114), enhance this approach with probabilistic elements, enabling the generation of new data samples. This project analyzes the impact of various parameters on these models' performance, focusing on reconstruction quality and compression efficiency.

## Dataset

The dataset used is the Flickr Faces 70k Thumbnails 128x128, containing 70,000 grayscale images of faces. The dataset is split into:
- **Train**: 70% (49,000 images)
- **Validation**: 10% (7,000 images)
- **Test**: 20% (14,000 images)

[Download the dataset here](https://www.kaggle.com/datasets/imcr00z/flickr-faces-70k-thumbnails-128x128).

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
| MSE      | 0.00426| 0.00420| 0.00085            |
| PSNR     | 23.989 | 24.020 | 1.258              |
| SSIM     | 0.72741| 0.73000| 0.051              |

*Table 1: AutoEncoder reconstruction quality metrics.*

#### Regularization Techniques

**Metrics Table for Regularization Techniques:**

| Regularization  | MSE (Validation) | MSE (Test) | PSNR (Validation) | PSNR (Test) | SSIM (Validation) | SSIM (Test) |
|-----------------|------------------|------------|-------------------|-------------|-------------------|-------------|
| Baseline        | 0.00426          | 0.00420    | 23.989 dB         | 24.020 dB   | 0.72741           | 0.73000     |
| L1              | 0.00895          | 0.00890    | 21.178 dB         | 21.200 dB   | 0.62132           | 0.62000     |
| L2              | 0.00442          | 0.00435    | 23.855 dB         | 23.880 dB   | 0.70792           | 0.71000     |
| L1 + L2         | 0.00450          | 0.00445    | 23.772 dB         | 23.790 dB   | 0.72140           | 0.72000     |

**Figures:**

1. ![Autoencoder Regularization Images](Figures/AE_reg_images2.png)  
   *Figure 2: Autoencoder regularization images showing the effect of different regularization techniques.*

2. ![Autoencoder Training Loss](Figures/AE_loss.png)  
   *Figure 3: Training loss over epochs for Autoencoder.*

#### Noise Robustness

The Autoencoder exhibited a decline in reconstruction quality with increasing noise levels. The MSE increased from `0.00426` to `0.00815` as Gaussian noise with a standard deviation of `0.1` was added. Regularization techniques, particularly L2 and Elastic Net, improved the robustness to noise, with Elastic Net providing a more stable reconstruction quality under noisy conditions.

**Figures:**

1. ![AE Noise Regularization Images](Figures/AE_noise_reg_images.png)  
   *Figure 4: Impact of noise and regularization on Autoencoder performance.*

### Variational Autoencoder (VAE)

![Variational Autoencoder (VAE) Architecture](https://lilianweng.github.io/posts/2018-08-12-vae/vae-gaussian.png)  
*Figure 5: Diagram of the Variational Autoencoder (VAE) Architecture. Source: [Lilian Weng's Blog](https://lilianweng.github.io/posts/2018-08-12-vae/vae-gaussian.png)*

- **Architecture**: Encoder with three linear layers (128x128 input to 512 dimensions, then to 200 dimensions for mean and log variance); Decoder with two linear layers (200 to 512 dimensions and 512 to 128x128 dimensions).
- **Activation Functions**: ReLU in the encoder, Sigmoid in the decoder.
- **Training**: Combination of reconstruction loss and KL divergence, optimized with Adam for 200 epochs.

#### Reconstruction Quality

**Metrics Table for VAE:**

| Metric   | Mean   | Median | Standard Deviation |
|----------|--------|--------|--------------------|
| MSE      | 0.03220| 0.03220| 0.00000            |
| PSNR     | 10.385 | 10.385 | 0.000              |
| SSIM     | 0.25626| 0.25626| 0.000              |

*Table 2: VAE reconstruction quality metrics.*

#### Regularization Techniques

**Metrics Table for Regularization Techniques in VAE:**

| Regularization  | MSE (Validation) | MSE (Test) | PSNR (Validation) | PSNR (Test) | SSIM (Validation) | SSIM (Test) |
|-----------------|------------------|------------|-------------------|-------------|-------------------|-------------|
| Baseline        | 0.03220          | 0.03030    | 10.385 dB         | 10.246 dB   | 0.25626           | 0.24562     |
| L1              | 0.03218          | 0.03100    | 10.374 dB         | 10.400 dB   | 0.25657           | 0.25000     |
| L2              | 0.04598          | 0.02700    | 7.395 dB          | 10.600 dB   | 0.21202           | 0.26000     |
| L1 + L2         | 0.06001          | 0.02300    | 5.883 dB          | 10.800 dB   | 0.19449           | 0.27000     |

**Figures:**

1. ![VAE Regularization Images](Figures/vae_reg_images_2.png)  
   *Figure 6: Variational Autoencoder regularization images demonstrating the effect of different regularization techniques.*

2. ![VAE Training Loss](Figures/VAE_train_loss.png)  
   *Figure 7: Training loss over epochs for VAE.*

3. ![VAE Validation Loss](Figures/VAE_val_loss.png)  
   *Figure 8: Validation loss over epochs for VAE.*

#### Noise Robustness

The VAE demonstrated better robustness to noise compared to the Autoencoder. The MSE increase with noise was less pronounced, going from `0.03220` to `0.03700` with Gaussian noise (σ=0.1). Regularization techniques, particularly Elastic Net, contributed to better handling of noisy inputs, with noticeable improvements in noise robustness.

**Figures:**

1. ![VAE Noise Regularization Images](Figures/VAE_noise_reg_images.png)  
   *Figure 9: Impact of noise and regularization on Variational Autoencoder performance.*

## Conclusion

Both VAE and AE architectures exhibit strengths and limitations in image compression tasks. The VAE shows a lower reconstruction quality with higher MSE and lower PSNR and SSIM compared to the AE. However, the VAE demonstrates better performance with regularization techniques, particularly Elastic Net, and shows better robustness to noise compared to the traditional Autoencoder. The MSE for VAE with noise is less pronounced compared to the AE, highlighting its superior performance in handling noisy inputs. Regularization techniques also play a crucial role in improving the stability and generalization of both models. Elastic Net regularization, in particular, showed notable improvements in noise robustness for both AE and VAE.

## References

1. Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114).
2. Bengio, Y., et al. (2006). Greedy Layer-Wise Training of Deep Networks. [Neural Computation](https://direct.mit.edu/neco/article/18/7/1536/6850/Greedy-Layer-Wise-Training-of-Deep-Networks).
3. Lilián Weng. (2018). A Comprehensive Introduction to Different Types of Autoencoders. [Lilian Weng's Blog](https://lilianweng.github.io/posts/2018-08-12-vae/).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
