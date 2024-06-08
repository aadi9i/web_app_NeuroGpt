import React, { useState, useEffect } from "react";
import Navbar from "../../components/Navbar/Navbar";
import axios from "axios";
import Plot from "react-plotly.js";
import { ThreeCircles } from "react-loader-spinner";
import "./ChannelZ.css";
import Sidebar from "../../components/Sidebar/Sidebar";
import Typewriter from "typewriter-effect";
import Clot from "../Clot/Clot";
import Video from "../../components/Video/Video";

const Channel = () => {
  const [plotData, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const sentences = [
    "Too much of this well-acted but dangerously slow thriller feels like a preamble to a bigger, more complicated story, one that never materializes.",
    "It's solid and affecting and exactly as thought-provoking as it should be.",
    "A richly imagined and admirably mature work from a gifted director who definitely has something on his mind.",
    "Co-writer/director Jonathan Parker's attempts to fashion a Brazil-like, hyper-real satire fall dreadfully short.",
    "It isn't that Stealing Harvard is a horrible movie -- if only it were that grand a failure!",
    "It's a head-turner -- thoughtfully written, beautifully read and, finally, deeply humanizing.",
    "It isn't that Stealing Harvard is a horrible movie -- if only it were that grand a failure!",
    "It just doesn't have much else... especially in a moral sense.",
    "Viewed as a comedy, a romance, a fairy tale, or a drama, there's nothing remotely triumphant about this motion picture.",
  ];
  const [selectedSentence, setSentence] = useState(sentences[0]);


  function getRandomSentence() {
    var randomIndex = Math.floor(Math.random() * 9);
    setSentence(sentences[randomIndex]);
  }

  const fetchData = async () => {
    console.log(selectedSentence);
    try {
      setLoading(true);
      const response = await axios.post("http://localhost:5000/channelz", {
        sentence: selectedSentence,
      });
      setData(response.data.data);
      console.log({ 1: response.data.data });
      setLoading(false);
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };
  const defaultOption = "HI";
  const [selectedOption, setSelectedOption] = useState(defaultOption);

  const handleSelect = (e) => {
    const selectedValue = e.target.value;
    setSelectedOption(selectedValue);
    setSentence(selectedValue);
  };

  return (
    <div className="styles.main">
      <Navbar />

      <div>
        <div className="plot">
          <div className="clot">
            <Clot /> 
          </div>

          <div className="video">
            <Video />
          </div>
        </div>

        <div className="box">
          {selectedSentence && (
            <div className="sentence">{selectedSentence}</div>
          )}

          <select value={selectedOption} onChange={handleSelect}>
            <option value="" disabled hidden>
              {defaultOption}
            </option>
            {sentences.map((option,index) => (
              <option key={index} value={option}>
                {option}
              </option>
            ))}
          </select>

          {loading ? (
            <div className="loader">
              <ThreeCircles
                height="50"
                width="50"
                color="#046cf1"
                wrapperStyle={{}}
                wrapperClass=""
                visible={true}
                ariaLabel="three-circles-rotating"
                outerCircleColor=""
                innerCircleColor=""
                middleCircleColor=""
              />
            </div>
          ) : (
            <div className="predict" onClick={fetchData}>
              <button class="btn btn-primary rounded-pill px-3" type="button">
                {plotData ? (
                  <Typewriter
                    options={{ strings: [plotData[0]], autoStart: true }}
                  />
                ) : (
                  "Lets's Predict Your Mind"
                )}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Channel;
