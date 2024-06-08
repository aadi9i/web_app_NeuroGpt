import React from "react";
import styles from "./Header.module.css";
import chatbotBanner from "../../assets/chatbotbanner.svg";
import { Link } from "react-router-dom";
import brain from "../../assets/brain.png";
import neuropgt from "../../assets/neuropgt.png" ;
const Header = () => {
  return (
    <div className={styles.container}>


      <div className={styles.left}>
        <p className={styles.heading}>
          "NeuroGPT: Decoding Human Intelligence"
        </p>
        <p className={styles.subHeading}>

        </p>
        <Link to="/chatbox">
          <button className={styles.btn}>Get Started</button>
        </Link>
      </div>
      <div className={styles.right}>
        <img src={neuropgt} style={{ width: '700px', height: '350px',marginLeft:"15px" }} alt="AI" />
      </div>
    </div>
  );
};

export default Header;
