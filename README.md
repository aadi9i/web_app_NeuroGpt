# EEG-to-Text Real-Time Web App for Communication Assistance

This project leverages EEG (electroencephalography) technology and powerful language models to decode brain signals into text in real time. Our goal is to provide a communication tool for individuals with severe communication impairments.

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
   npm install         # Installs main project dependencies
   cd frontend
   npm install         # Installs frontend dependencies
   cd ..
3. **Start your frontend :**
   ```bash
   npm start 
4. **Start your backend :**
   ```bash
   cd backend
   python neuro1.py    