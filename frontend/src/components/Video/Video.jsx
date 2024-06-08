import React from 'react'
import brainvideo from "./brainvideo.mp4"
import "./Video.css";

const Video = () => {
    return (
      <div className>
        <video className='mp' src={brainvideo} autoPlay loop muted />
          
      </div>
    );
  };
  
  export default Video;