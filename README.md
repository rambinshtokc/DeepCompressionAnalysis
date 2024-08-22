# Face Image Compression

## Introduction

The Face Image Compression project investigates the use of deep learning models, specifically Autoencoders (AE) and Variational Autoencoders (VAE), for compressing and reconstructing face images. This project aims to demonstrate the effectiveness of these models in reducing image size while preserving quality and compare their performance.

## Project Goal

The primary goal of this project is to explore and evaluate the capabilities of AE and VAE models for face image compression. We aim to:

- Develop and implement AE and VAE models for image compression.
- Compare the performance of these models in terms of image quality and compression efficiency.
- Provide a clear understanding of how these models work and their practical applications.

## Method

### Autoencoder (AE)

The AE model is designed to compress face images into a lower-dimensional latent space and then reconstruct them. It consists of:

- **Encoder**: Compresses the input image into a lower-dimensional representation.
- **Decoder**: Reconstructs the image from the compressed representation.

### Variational Autoencoder (VAE)

The VAE model extends the AE by introducing probabilistic layers. It includes:

- **Encoder**: Produces a probability distribution over the latent space instead of a fixed vector.
- **Decoder**: Samples from this distribution to reconstruct the image.

### Training

Both models are trained on face image datasets using the following steps:

1. **Preprocessing**: Resizing and normalizing images.
2. **Training**: Using reconstruction loss and, for VAE, additional KL divergence loss.
3. **Evaluation**: Assessing image quality through visual inspection and quantitative metrics.

## Experiments and Results

### Autoencoder

- **Architecture**: A basic AE with fully connected layers.
- **Results**: Demonstrated reasonable reconstruction quality with moderate compression.

### Variational Autoencoder

- **Architecture**: VAE with probabilistic latent variables.
- **Results**: Provided improved reconstruction quality and regularization compared to AE, with enhanced handling of variability in the data.

### Comparison

- **Image Quality**: VAE generally produced higher-quality reconstructions with better handling of complex features.
- **Compression Efficiency**: Both models achieved significant compression, but VAE showed more robustness in retaining image details.

## Conclusions

The project confirmed that both AE and VAE are effective for face image compression. VAE, with its probabilistic approach, outperformed AE in terms of image quality and robustness. These findings suggest that VAE is a promising model for practical image compression applications.

## Future Work

- **Model Enhancement**: Explore advanced architectures and techniques to further improve compression efficiency and image quality.
- **Data Variability**: Test models on diverse datasets to evaluate generalization capabilities.
- **Real-world Applications**: Investigate the deployment of these models in real-world scenarios, such as mobile or web applications.

## How to Run

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/rambinshtokc/face-image-compression.git
cd face-image-compression

2. Set Up Your Environment
Create and activate a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required packages:

bash
Copy code
pip install -r requirements.txt
3. Run the Notebooks
Launch Jupyter Notebook and open the notebooks:

bash
Copy code
jupyter notebook AE.ipynb
jupyter notebook VAE.ipynb
Follow the instructions in each notebook to run the experiments and view results.

Ethics Statement
This project adheres to ethical guidelines in machine learning and data science. We ensure that all data used is anonymized and handled with confidentiality. The models and methods presented aim to advance knowledge and applications in image compression while respecting privacy and ethical considerations.
