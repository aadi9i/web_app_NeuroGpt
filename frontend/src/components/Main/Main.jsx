import React from "react";
import styles from "./Main.module.css";
import ChatBotCardImg from "../../assets/Chat-bot-bro.svg";
import ResponsiveImg from "../../assets/responsive.svg";
import ConversationalImg from "../../assets/conversational.jpg";
import non_invasive from "../../assets/non_invasive.png";
import stephen from "../../assets/stephen.png";
import { Link } from "react-router-dom";
import scalp from "../../assets/scalp.png";
const Main = () => {
  return (
    <div className={styles.container}>
      <div className={styles.heading}>
        <h3>Features</h3>
      </div>
      <div className={styles.cards}>
        <div className={styles.card}>
          <div className={styles.image}>
            <img src={stephen} style ={{width:"200px"}} alt="ConversationalImg" />
          </div>
          <div className={styles.text}>
            <h1 className={styles.cardTitle}>Decoding Thoughts</h1>
            <p>
            Traditionally, individuals with paralysis or vocal anomalies and with severe communication impairments have limited ways to express themselves. Our NeuroGPT- a LLM powered deep learning model aims to revolutionize Assistive Technologies by harnessing brain signals and translating them directly into a continous text.

            </p>
          </div>
        </div>
        <div className={styles.card}>
          <div className={styles.image}>
            <img src={non_invasive} alt="non invasive" />
          </div>
          <div className={styles.text}>
            <h1 className={styles.cardTitle}>Non-Invasive Brain Imaging: EEG</h1>
            <p>
            Our brains are extraordinary communication centers, processing information, feelings, and intentions through complex electrical signals. EEG (electroencephalography) technique allows us to non-invasively capture these signals, which reflects the practicality of this method compared to the existing invasive methods.


            </p>
          </div>
        </div>
        <div className={styles.card}>
          <div className={styles.image}>
            <img src={scalp} style ={{height:"300px"}} alt="ResponsiveImg" />
          </div>
          <div className={styles.text}>
            <h1 className={styles.cardTitle}>Adaptive Channel Configuration</h1>
            <p>
            Our approach prioritizes both user convenience and research flexibility as it supports realtime data from the 4 EEG channels primarily associated with language processing. Additionally, we provide comprehensive support for 128-channel EEG devices, allowing for high-resolution analysis and the potential to explore the contributions of other brain regions to the language decoding process.
            </p>
          </div>
        </div>
      </div>

    </div>
  );
};

export default Main;
