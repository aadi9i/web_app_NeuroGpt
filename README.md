# EEG-to-Text Decoding using LLMs and Real-Time Web App for Communication Assistance

Our project leverages EEG (electroencephalography) technology and powerful large language models to decode brain signals into  text as well implement it in real time. Our goal is to provide a communication tool for individuals with severe communication impairments.

## Research Paper Implementation drive link [https://github.com/alexandrebarachant/muse-lsl](https://drive.google.com/drive/folders/1Pep7mpqO65n41xJj0R9teEq9ex8wlekE?usp=drive_link](https://github.com/alexandrebarachant/muse-lsl)

## Getting Started

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
