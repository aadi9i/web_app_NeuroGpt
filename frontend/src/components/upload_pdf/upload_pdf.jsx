import React, { useState,useRef } from 'react';
import { Upload as UploadIcon } from 'react-feather';
import axios from 'axios'; // You'll need to install axios

function PdfUploadButton() {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async (event) => {
    event.preventDefault();
    fileInputRef.current.click();

    

    const formData = new FormData();
    formData.append('pdfFile', selectedFile);

    try {
      const response = await axios.post('/api/upload-pdf', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      console.log('Upload response:', response); 
      alert('File uploaded successfully!');
      setSelectedFile(null); // Reset for next upload
    } catch (error) {
      console.error('Upload error:', error);
      alert('An error occurred during upload.');
    }
  };
  const fileInputRef = useRef(null);

  return (
    <div className='upload'>
<div>
      <input 
        type="file" 
        accept=".pdf" 
        onChange={handleFileChange} 
        ref={fileInputRef} 
        style={{ display: 'none' }} 
      /> 
      <button onClick={handleUpload}>
        <UploadIcon /> Upload PDF
      </button>
    </div>
  </div>
  );
}

export default PdfUploadButton;