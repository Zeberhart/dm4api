import './App.css';
import Chatbox from "./Chatbox"

import typingAnim from "./img/typing.gif"
import logo from "./img/newlogo.png"

import React, { useState, useEffect, useContext, useRef} from 'react';
import ReactDOM from 'react-dom';

import Divider from '@material-ui/core/Divider';
import Paper from '@material-ui/core/Paper';
import Tooltip from '@material-ui/core/Tooltip';
import InputBase from '@material-ui/core/InputBase';
import TextField from '@material-ui/core/TextField';
import InputAdornment from "@material-ui/core/InputAdornment";
import IconButton from "@material-ui/core/IconButton";
import Button from "@material-ui/core/Button";
import MuiAccordion from '@material-ui/core/Accordion';
import MuiAccordionSummary from '@material-ui/core/AccordionSummary';
import MuiAccordionDetails from '@material-ui/core/AccordionDetails';
import Collapse from '@material-ui/core/Collapse';
import Switch from '@material-ui/core/Switch';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import SearchIcon from "@material-ui/icons/Search";
import ListAltIcon from "@material-ui/icons/ListAlt";
import AddIcon from "@material-ui/icons/Add";
import AndroidIcon from "@material-ui/icons/Android";
import FaceIcon from "@material-ui/icons/Face";
import CachedIcon from "@material-ui/icons/Cached";
import HelpOutlineIcon from "@material-ui/icons/HelpOutline";
import { makeStyles } from '@material-ui/core/styles';



import {PopIn} from "react-spring-pop";
var shuffleSeed = require('shuffle-seed');


function App() {

  const [messages, setMessages] = useState([]);
  const [policy, setPolicy] = useState("hc");
  const [taskset, setTaskset] = useState("libssh");
  
  function _addMessage(message){
    setMessages(oldMessages => [...oldMessages, message])
  } 



  return (
    <div>
      <div className="vh-100 container-fluid px-4 py-3">
        <div className="h-100 d-flex justify-content-center">

          <div className="h-100 mx-4 px-0 py-0 d-flex align-items-center justify-content-center">
          <div className="d-flex flex-column me-5">
            <div className="d-flex flex-column ">
              <h2>Policy</h2>
              <Button style={{margin:"5px", backgroundColor:(policy=="hc"?"#bbffc0":"initial")}} onClick={()=>setPolicy("hc")} variant="outlined">
                Hand-crafted
              </Button>
              <Button style={{margin:"5px", backgroundColor:(policy=="rl"?"#bbffc0":"initial")}} onClick={()=>setPolicy("rl")} variant="outlined">
                Learned
              </Button>
              <Button style={{margin:"5px", backgroundColor:(policy=="search"?"#bbffc0":"initial")}} onClick={()=>setPolicy("search")} variant="outlined">
                Baseline
              </Button>
            </div>
            <br/>
            <div className="d-flex flex-column ">
              <h2>API dataset</h2>
              <Button style={{margin:"5px", backgroundColor:(taskset=="libssh"?"#99eeb0":"#dddddd")}} onClick={()=>{setMessages([]); setTaskset("libssh")}} variant="contained" disableElevation>
                Libssh
              </Button>
              <Button style={{margin:"5px", backgroundColor:(taskset=="allegro"?"#99eeb0":"#dddddd")}} onClick={()=>{setMessages([]); setTaskset("allegro")}} variant="contained" disableElevation>
                Allegro
              </Button>
            </div>
          </div>
            <Chatbox 
              addMessage={_addMessage} 
              messages={messages} 
              policy={policy} 
              taskset={taskset}
            />
          </div>
        </div>
      </div>
    </div>
  );
}












export default App;
