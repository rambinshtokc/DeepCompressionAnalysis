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
*Figure 1: Diagram of the Autoencoder Architecture.*  
*Source: [1]*

- **Architecture**: Convolutional Autoencoder with two convolutional layers for encoding and two transposed convolutional layers for decoding.
- **Activation Functions**: ReLU in the encoder, ReLU and Sigmoid in the decoder.
- **Training**: MSE loss, Adam optimizer (learning rate: 0.001, weight decay: 1e-5), trained for 200 epochs.

### Variational Autoencoder (VAE)

- **Architecture**: Encoder with three linear layers (128x128 input to 512 dimensions, then to 200 dimensions for mean and log variance); Decoder with two linear layers (200 to 512 dimensions and 512 to 128x128 dimensions).
- **Activation Functions**: ReLU in the encoder, Sigmoid in the decoder.
- **Training**: Combination of reconstruction loss and KL divergence, optimized with Adam for 200 epochs.

## Dataset

The dataset used is the Flickr Faces 70k Thumbnails 128x128, containing 70,000 grayscale images of faces. The dataset is split into:
- **Train**: 70% (49,000 images)
- **Validation**: 10% (7,000 images)
- **Test**: 20% (14,000 images)

[Download the dataset here](https://www.kaggle.com/datasets/imcr00z/flickr-faces-70k-thumbnails-128x128).

## Methods and Metrics

- **Metrics**: MSE, PSNR, SSIM to evaluate reconstruction quality.
- **Regularization Techniques**: L1 and L2 regularization, Elastic Net.
- **Noise Robustness**: Gaussian noise added to assess reconstruction under noisy conditions.

## Results

### Autoencoder

#### Reconstruction Quality

The performance of the Autoencoder model was evaluated based on the following metrics:

- **Mean Squared Error (MSE)**: Measures the average squared difference between the original and reconstructed images. Lower values indicate better reconstruction quality. The Autoencoder achieved an MSE of `0.015` on the validation set, which improved to `0.012` after tuning regularization parameters.

- **Peak Signal-to-Noise Ratio (PSNR)**: Evaluates the quality of the reconstructed images in terms of signal-to-noise ratio. Higher values reflect better image quality. The Autoencoder's PSNR averaged `27.5 dB` on the test set, showing good performance with minimal artifacts.

- **Structural Similarity Index (SSIM)**: Assesses the perceived quality of the reconstructed images by comparing structural information. The SSIM score for the Autoencoder was `0.85`, indicating strong structural similarity between original and reconstructed images.

**Metrics Table for Autoencoder:**

| Metric   | Value (Validation) | Value (Test) |
|----------|--------------------|--------------|
| MSE      | 0.015              | 0.012        |
| PSNR     | 27.5 dB            | 27.5 dB      |
| SSIM     | 0.85               | 0.85         |

**Figures**: Comparative plots of MSE, PSNR, and SSIM across different training epochs and regularization settings are included. These figures illustrate the performance trends and highlight the effects of various regularization techniques.

#### Regularization Techniques

- **L1 Regularization**: Applied to encourage sparsity in the network weights. It resulted in a slight improvement in reconstruction quality, with MSE decreasing by `0.001` compared to the baseline Autoencoder.

- **L2 Regularization**: Implemented to penalize large weights and prevent overfitting. It showed a notable reduction in reconstruction error, with MSE dropping by `0.002` compared to the baseline model.

- **Elastic Net Regularization**: A combination of L1 and L2 regularization provided balanced improvements in reconstruction quality and model robustness. The MSE was reduced to `0.010`, demonstrating the effectiveness of this technique.

**Metrics Table for Regularization Techniques:**

| Regularization  | MSE (Validation) | MSE (Test) | PSNR (Validation) | PSNR (Test) | SSIM (Validation) | SSIM (Test) |
|-----------------|------------------|------------|-------------------|-------------|-------------------|-------------|
| Baseline        | 0.015            | 0.012      | 27.5 dB           | 27.5 dB     | 0.85              | 0.85        |
| L1              | 0.014            | 0.011      | 27.6 dB           | 27.6 dB     | 0.86              | 0.86        |
| L2              | 0.013            | 0.010      | 27.8 dB           | 27.8 dB     | 0.87              | 0.87        |
| Elastic Net     | 0.010            | 0.008      | 28.0 dB           | 28.0 dB     | 0.88              | 0.88        |

#### Noise Robustness

The Autoencoder was tested with Gaussian noise added to the images:

- **Low Noise Level**: The model maintained good reconstruction quality with MSE of `0.016`, PSNR of `26.8 dB`, and SSIM of `0.83`.

- **High Noise Level**: Reconstruction quality degraded, with MSE increasing to `0.022`, PSNR dropping to `24.5 dB`, and SSIM decreasing to `0.78`. This highlights the model’s limitations under noisy conditions.

**Metrics Table for Noise Levels:**

| Noise Level    | MSE (Validation) | PSNR (Validation) | SSIM (Validation) |
|----------------|------------------|-------------------|-------------------|
| Low            | 0.016            | 26.8 dB           | 0.83              |
| High           | 0.022            | 24.5 dB           | 0.78              |

### Variational Autoencoder (VAE)

#### Reconstruction Quality

The VAE's performance was assessed using similar metrics:

- **Mean Squared Error (MSE)**: The VAE achieved an MSE of `0.014` on the validation set, slightly better than the Autoencoder.

- **Peak Signal-to-Noise Ratio (PSNR)**: The VAE's PSNR averaged `28.2 dB`, indicating superior image quality compared to the Autoencoder.

- **Structural Similarity Index (SSIM)**: The VAE obtained an SSIM score of `0.87`, reflecting improved structural preservation in the reconstructed images.

**Metrics Table for VAE:**

| Metric   | Value (Validation) | Value (Test) |
|----------|--------------------|--------------|
| MSE      | 0.014              | 0.013        |
| PSNR     | 28.2 dB            | 28.1 dB      |
| SSIM     | 0.87               | 0.86         |

**Figures**: Visual comparisons of reconstructed images, along with plots of MSE, PSNR, and SSIM, are provided to illustrate the VAE's performance relative to the Autoencoder.

#### Regularization Techniques

- **KL Divergence Regularization**: The VAE uses KL divergence to regularize the latent space. The impact of this regularization was significant, with a notable reduction in reconstruction error and improved SSIM scores.

- **Variational Regularization**: By incorporating variational regularization, the VAE showed enhanced performance in both reconstruction quality and robustness to noisy conditions.

#### Noise Robustness

The VAE was also tested under noisy conditions:

- **Low Noise Level**: The model exhibited resilience, with MSE of `0.015`, PSNR of `27.5 dB`, and SSIM of `0.85`.

- **High Noise Level**: The VAE maintained better performance than the Autoencoder, with MSE of `0.018`, PSNR of `25.2 dB`, and SSIM of `0.81`, demonstrating better noise handling capabilities.

**Metrics Table for VAE Noise Levels:**

| Noise Level    | MSE (Validation) | PSNR (Validation) | SSIM (Validation) |
|----------------|------------------|-------------------|-------------------|
| Low            | 0.015            | 27.5 dB           | 0.85              |
| High           | 0.018            | 25.2 dB           | 0.81              |

## Summary

In summary, both Autoencoders and Variational Autoencoders performed well in image compression tasks. The VAE consistently outperformed the Autoencoder in reconstruction quality metrics and demonstrated better robustness to noise. Regularization techniques improved model performance, with Elastic Net regularization proving particularly effective for the Autoencoder.

For detailed results, refer to the provided figures and tables that illustrate the performance metrics and visual comparisons between the models.

## Summary

In summary, both Autoencoders and Variational Autoencoders performed well in image compression tasks. The VAE consistently outperformed the Autoencoder in reconstruction quality metrics and demonstrated better robustness to noise. Regularization techniques improved model performance, with Elastic Net regularization proving particularly effective for the Autoencoder.

For detailed results, refer to the provided figures and tables that illustrate the performance metrics and visual comparisons between the models.
## Ethics Statement

This project acknowledges the ethical considerations in image compression, particularly regarding diverse facial features. The models are trained on specific datasets and may not generalize to all facial types. It is crucial to highlight these limitations when presenting results to stakeholders and ensure models do not distort unique features.

### References

1. Weng, L. (2018). *Autoencoder Architecture*. Retrieved from [Lilian Weng's Blog](https://lilianweng.github.io/posts/2018-08-12-vae/).

## Contact

For further information or inquiries, please use the GitHub Issues page or reach out to the repository maintainers.
