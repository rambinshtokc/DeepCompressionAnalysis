Performance Analysis of VAE and Autoencoder Architectures
Overview
This project performs a comprehensive analysis of Variational Autoencoders (VAE) and traditional Autoencoders (AE) to evaluate their performance in image compression tasks. The study focuses on the impact of various model parameters and configurations on key performance metrics, including reconstruction quality, compression ratio, and noise robustness.

Authors
Ram Binshtock: rambinshtock@campus.technion.ac.il
Sahar Zeltzer: saharzeltzer@campus.technion.ac.il
Abstract
This study investigates how different configurations and parameters affect the performance of image compression using deep neural network architectures, specifically Variational Autoencoders (VAE) and traditional Autoencoders. Metrics such as Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM) are used to assess reconstruction quality. Various aspects including regularization techniques (L1, L2), noise robustness, and more are analyzed. The findings aim to guide the optimization of VAE and AE architectures for improved image compression.

Introduction
Autoencoders and Variational Autoencoders are crucial tools for dimension reduction and image compression. Autoencoders, introduced in 2006, learn efficient data representations by compressing input data into a latent space and reconstructing it. VAEs, introduced in 2013, enhance this approach with probabilistic elements, enabling the generation of new data samples. This project analyzes the impact of various parameters on these models' performance, with a focus on reconstruction quality and compression efficiency.

Baseline Architectures
Autoencoder
Architecture: A convolutional Autoencoder with two convolutional layers for encoding and two transposed convolutional layers for decoding.
Activation Functions: ReLU activations in the encoder, ReLU and Sigmoid activations in the decoder.
Training: Mean Squared Error (MSE) loss, Adam optimizer with a learning rate of 0.001, weight decay of 1e-5, trained for 200 epochs.
Variational Autoencoder (VAE)
Architecture: The encoder consists of three linear layers (128x128 input to 512 dimensions, then to 200 dimensions for mean and log variance). The decoder consists of two linear layers (200 to 512 dimensions and 512 to 128x128 dimensions).
Activation Functions: ReLU activations in the encoder, Sigmoid activation in the decoder.
Training: A combination of reconstruction loss and KL divergence, optimized with Adam for 200 epochs.
Dataset
The dataset used is the Flickr Faces 70k Thumbnails 128x128, containing 70,000 grayscale images of faces. The dataset is split into:

Train: 70% (49,000 images)
Validation: 10% (7,000 images)
Test: 20% (14,000 images)
Download the dataset here.

Methods and Metrics
Metrics: MSE, PSNR, SSIM to evaluate reconstruction quality.
Regularization Techniques: L1 and L2 regularization, Elastic Net.
Noise Robustness: Gaussian noise added to assess reconstruction under noisy conditions.
Results
Autoencoder
Reconstruction Quality: Detailed results of MSE, SSIM, and PSNR are provided in tables and figures.
Regularization Types: Impact on loss function and reconstructed images shown.
Noise Robustness: Effect of Gaussian noise on reconstruction quality and images demonstrated.
Variational Autoencoder
Reconstruction Quality: Comparative analysis of reconstruction metrics.
Regularization Types: Impact on loss function and generated images.
Noise Robustness: Analysis of model performance under noisy conditions.
Ethics Statement
This project acknowledges the ethical considerations in image compression, particularly regarding diverse facial features. The models are trained on specific datasets and may not generalize to all facial types. It is crucial to highlight these limitations when presenting results to stakeholders and ensure models do not damage unique features, such as tattoos.

Contact
For further information or inquiries, please contact:

Ram Binshtock: rambinshtock@campus.technion.ac.il
Sahar Zeltzer: saharzeltzer@campus.technion.ac.il
