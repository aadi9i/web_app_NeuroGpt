import React, { useEffect, useState } from "react";
import Plot from 'react-plotly.js';
import styles from "./ChatBox.module.css";
import Title from "../../components/Title/Title";
import InputBar from "../../components/InputBar/InputBar";

import Body from "../../components/Body/Body";
import { ThreeCircles } from "react-loader-spinner";
import { auth } from "../../firebase";
import { signOut } from "firebase/auth";
import { Link } from "react-router-dom";
import Navbar from "../../components/Navbar/Navbar";
import axios from "axios";
import channel_img from "../../assets/channel_img.png";
import channel128_img from "../../assets/channel128_img.png";
import Clot from "../Clot/Clot";
import Channel from "../Channel/Channel"
import ChannelZ from "../ChannelZ/ChannelZ";
const ChatBox = (props) => {
  const [data, setData] = useState({});
  // const startStream = async () => {
  //   try {
  //     const response = await axios.get('http://localhost:5000/start_stream');
  //     // setData(response.data);
  //     // console.log(response.data);
  //   } catch (error) {
  //     console.error('Error fetching data:', error);
  //   }
  // };
  // const fetchData = async () => {
  //   try {
  //     const response = await axios.get('http://localhost:5000/fetch_stream');
  //     setData(response.data);
  //     console.log(response.data);
  //   } catch (error) {
  //     console.error('Error fetching data:', error);
  //   }
  // };
  const [userName, setUserName] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
    }, 2000);

    auth.onAuthStateChanged((user) => {
      // console.log(user);
      if (user) {
        setUserName(user.displayName);
      } else {
        setUserName("");
      }
    });
  }, []);
  // const handleButtonClick = async () => {
    // try {
    //   const response = await axios.post('http://127.0.0.1:5000');
    //   console.log(response.data);
    // } catch (error) {
    //   console.error('Error executing Python script:', error);
    // }
    // startStream();
    // const intervalId=setInterval(fetchData,5000);
    // return () => clearInterval(intervalId);
  // };
  const signOutHandler = () => {
    signOut(auth)
      .then(() => {
        alert("Sign out Successfully!");
      })
      .catch((err) => {
        console.log(err.message);
      });
  };
  


  return (
    <>
   <div className="styles.main">
    <Navbar/>
   </div>
   <div>
   <div className="container py-4">
       

    <div className="p-5 mb-4 bg-light rounded-3">
      <div className="container-fluid py-5">
        <h2 className="display-5 fw-bold">Go for 128 Channels</h2>
        <p className="col-md-8 fs-4"><img src={channel128_img} style={{ width: '200px', height: '150px',marginLeft:"15px" }} alt="" /></p>
        <Link to = "/channelz" activeClassName="active" className="let"><button className="btn btn-primary btn-lg custom" type="button">Get Started</button></Link>
      </div>
    </div>
    <div className="p-5 mb-4 bg-light rounded-3">
      <div className="container-fluid py-5">
        <h2 className="display-5 fw-bold">Go for 4 Channels</h2>
        <p className="col-md-8 fs-4"><img src={channel_img} style={{ width: '200px', height: '150px',marginLeft:"15px" }} alt="" /></p>
        <Link to = "/channel" activeClassName="active"><button className="btn btn-primary btn-lg custom" type="button">Get Started</button></Link>
      </div>
    </div>
    <div className="p-5 mb-4 bg-light rounded-3">
      <div className="container-fluid py-5">
        <h2 className="display-5 fw-bold">See Real Time Plot</h2>
        
        <Link to = "/clot" activeClassName="active"><button className="btn btn-primary btn-lg custom" type="button">Get Started</button></Link>
      </div>
    </div>       


    </div>


  </div>
  
  <div></div>
   
  

    </>
  );
};

export default ChatBox;
