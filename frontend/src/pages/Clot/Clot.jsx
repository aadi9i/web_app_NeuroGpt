import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import Navbar from "../../components/Navbar/Navbar";
const Clot = () => {
  const [plotData, setData] = useState([]);
  
  useEffect(() => {
    const setStream = async () => {
        try {
          const response = await axios.get('http://localhost:5000/startstream');
        } catch (error) {
          console.error('Error fetching data:', error);
        }
      };
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/fetchstream');
        const newData = response.data.data;
        console.log(newData)
        setData(newData);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };
    
    const intervalId = setInterval(fetchData, 100); // Fetch data every 1 second
    setStream();
    return () => clearInterval(intervalId); // Clean up interval on unmount
  }, []);

  return (
    <div className="container mt-5">
 
      <div className="row">
        <div className="col-12">
          <h1 className="text-center">Real-Time EEG Recording</h1>
        </div>
      </div>
      <div className="row">
        <div className="col-md-8 offset-md-2">
        {plotData.map((plot, index) => (
            <div key={index}>
            {/* <h2 className="text-center">{plot.layout.title}</h2> */}
            <Plot data={plot.data} layout={plot.layout} />
            </div>
        ))}
        </div>
      </div>
    </div> 
  );
};

export default Clot;