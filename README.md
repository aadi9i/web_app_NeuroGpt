# EEG-to-Text Decoding using LLMs and Real-Time Web App for inference

Our project leverages EEG (electroencephalography) technology and powerful large language models (LLMs) to decode brain signals into text and implement this in real-time. Our goal is to provide a communication tool for individuals with severe communication impairments.

## Research Paper Implementation 
We have implemented a research paper on Semantic-aware Contrastive Learning (C-SCL) for EEG-to-text decoding. Due to the large size of the model weights, they are available via a [Google Drive link](https://drive.google.com/drive/folders/1Pep7mpqO65n41xJj0R9teEq9ex8wlekE?usp=drive_link) instead of directly uploading here.

### Approach
Our approach involves using a preencoder trained with C-SCL to transform EEG signals into a representation that can be effectively processed by a pretrained BART model to generate natural language text.

### Key Components
- **C-SCL Preencoder**: The C-SCL technique is applied to train the preencoder, which helps in mapping EEG signals to a latent space that is semantically aware and suitable for language decoding.
- **Pretrained BART Model**: We utilize a pretrained BART model, a transformer-based sequence-to-sequence model, to generate text from the latent representations produced by the preencoder.

### Implementation Details
- **Preencoder Weights**: The trained preencoder weights, obtained using different labeled training configurations, are available in the `saved_models_cscl` folder.
- **Complete Model Weights**: The full model weights, including those for the BART model and the preencoder, can be found in the `src/checkpoints` directory. This directory contains the best checkpoints from the entire training process as well as the final model state.

### Directory Structure
- **saved_models_cscl**: Contains the weights of the preencoder trained with various configurations.
- **src/checkpoints**: Includes the complete model checkpoints. You can find the best checkpoints after the entire training or the final model weights in this directory.


## Getting Started with the web app

### Prerequisites

* Python (3.7 or higher recommended)
* Node.js and npm (or yarn)
* Muselsl (EEG streaming library): Install following their guide: [https://github.com/alexandrebarachant/muse-lsl](https://github.com/alexandrebarachant/muse-lsl)
* A compatible EEG device that can stream data using Muselsl

### Installation & Setup

1. **Download the Project:** Clone or download this repository to your local machine.

2. **Install Dependencies:**
   ```bash
   cd <project_directory>
   npm install         #this installs main project dependencies
   cd frontend
   npm install         #this then installs frontend dependencies
   cd ..
3. **Start your frontend :**
   ```bash
   npm start 
4. **Start your backend :**
   ```bash
   cd backend
   python neuro1.py    
