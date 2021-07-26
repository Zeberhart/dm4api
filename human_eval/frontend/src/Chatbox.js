import typingAnim from "./img/typing.gif"
import logo from "./img/newlogo.png"

import React, { useState, useEffect, useContext, useRef} from 'react';
import ReactDOM from 'react-dom';

import TextareaAutosize from 'react-textarea-autosize';

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
import useAutocomplete from "@material-ui/lab/useAutocomplete";
import styled from "styled-components";

import SearchIcon from "@material-ui/icons/Search";
import ListAltIcon from "@material-ui/icons/ListAlt";
import AddIcon from "@material-ui/icons/Add";
import AndroidIcon from "@material-ui/icons/Android";
import FaceIcon from "@material-ui/icons/Face";
import RefreshIcon from '@material-ui/icons/Refresh';
import HelpOutlineIcon from "@material-ui/icons/HelpOutline";
import CheckIcon from "@material-ui/icons/Check";
import CloseIcon from "@material-ui/icons/Close";
import CancelIcon from '@material-ui/icons/Cancel';
import { makeStyles } from '@material-ui/core/styles';

import {PopIn} from "react-spring-pop";

import { config } from './config.js'


var randomColor = require('randomcolor'); 

function Chatbox(props){

  const ws = useRef(null);
  var num = 0

  const [typing, setTyping] = useState(0);
  const [functions, setFunctions] = useState(0);
  const [input, setInput] = useState("");
  const [keywords, setKeywords] = useState([]);
  const [connected, setConnected] = useState(false);
  const [newData, setNewData] = useState(null);

  function processMessage(message){
    const data = JSON.parse(message.data)
    setNewData(data)
  }

  function sendMessage(type, content={}){
    const message = JSON.stringify({type:type, ...content})
    console.log(content)
    ws.current.send(message)
    addMessage("user", type, content)
  }

  function addMessage(speaker, type, content={}){
    const newMessage = {speaker:speaker, type:type, ...content}
    props.addMessage(newMessage)
  }

  function _fetchFunctions(){
    fetch(config.url.FUNCTIONS+props.taskset)
      .then(response => response.json())
      .then(data=>setFunctions(data))
      .catch((error) => {
        console.log(error)
      });
  }

  function _start(){
    const message = JSON.stringify({type:"start", policy:props.policy, library:props.taskset})
    ws.current.send(message)
  }

  function _restart(){
    const message = JSON.stringify({type:"restart"})
    ws.current.send(message)
  }

  useEffect(() => {
    if(!newData || !("type" in newData)){
      console.log("Received mystery message:")
      console.log(newData)
    }else{
      if(newData["type"] == "typing"){
        setTyping(typing+1)
      }else{
        setTyping(Math.max(0,typing-1))
        addMessage("System", newData["type"], newData)
      }
    }
  }, [newData]);
  

  useEffect(() => {
      _fetchFunctions()
      ws.current = new WebSocket(config.url.WS);
      ws.current.onopen = () => {
        console.log("ws open");
        setConnected(true)
        _start()
      };
      ws.current.onclose = () => {
        console.log("ws closed");
        setConnected(false)
      }
      ws.current.onmessage = message => {
          processMessage(message)
      };
      return () => {
          ws.current.close();
      };
  }, []);

  useEffect(() => {
      if (connected){
        _fetchFunctions()
        _start()
      }
  }, [props.policy, props.taskset]);

  return (
    <FunctionsContext.Provider value={functions}>
      <Paper 
        className="px-0 d-flex flex-column align-items-center position-relative" 
        style={{height:"700px", width:"420px", opacity:1, pointerEvents:"auto"}} 
        elevation={3} 
        variant="outlined"
      >
        <div className="position-absolute d-flex flex-column" style={{right:15, top:10}}>
          <IconButton size="medium" onClick={()=>sendMessage("help", {text:"Help"})}>
            <HelpOutlineIcon />
          </IconButton>
          <IconButton size="medium" onClick={()=>sendMessage("restart", {text:"Restart dialogue"})}>
            <RefreshIcon />
          </IconButton>
        </div>
        <Messages sendMessage={sendMessage} setInput={setInput} messages={props.messages} typing={typing} />
        <div className="w-100 border-top d-flex flex-column align-items-center">
          <Searchbar input={input} taskset={props.taskset} setInput={setInput} keywords={keywords} setKeywords={setKeywords} sendMessage={sendMessage} />
        </div>
      </Paper>
    </FunctionsContext.Provider>
  );
}



function RLSwitch(props){
  const [usingRL, setUsingRL] = useState(true)
  const _handleChange = (event) =>{
    setUsingRL(event.target.checked)
    props.sendMessage("set-rl", {value:event.target.checked})
  }

  return(
    <Tooltip title={usingRL?"Use Basic policy":"Use RL Policy"}>
    <FormControlLabel
      control={<Switch/>}
      checked={usingRL}
      onChange={_handleChange}
      label={"RL"}
      className=" mx-0"
    />
    </Tooltip>
  )
}

const FunctionsContext = React.createContext({})

function Messages(props){

  const [scrolled, setScrolled] = useState(false);
  const refContainer = useRef(null);

  useEffect(() => {
    if(!scrolled){
      refContainer.current.scrollTop = refContainer.current.scrollHeight;
    }
  }, [props.messages, props.typing]);

  return(
    <div className="w-100 h-100 pt-3 pb-4 flex-fill overflow-auto d-flex flex-column align-items-center" 
      style={{scrollBehavior: "smooth"}} 
      ref={refContainer}
    >
      <div className="d-flex flex-column col-11" >
        {props.messages.map((message, index)=>
            <Message 
              key={index} 
              sendMessage={props.sendMessage} 
              setInput={props.setInput}
              latest={props.speaker !== "user" && index === props.messages.length-1}
              {...message}
            />
        )}
        {props.typing>0 && 
          <PopIn friction={15}>
            <Message type={"typing"} />
          </PopIn>
        }
      </div>
    </div>
  )
}

function Message(props){
  const alignment = (props.speaker === "user" ? "align-self-end" : "align-self-start")
  const color = (props.speaker === "user" ? "#38FaD5" : "#d4d8dd")
  const icon = (props.speaker === "user" ? 
    <div className={"order-3"}>
      <FaceIcon fontSize="large" style={{color:"#0f5245"}}/>
    </div> 
  : 
    <div className={"order-0"}>
      <AndroidIcon fontSize="large"/>
    </div>
  )

  if(props.type == "RESTART"){
    return(
      null
    )
  }else{
    return(
      <div 
        className={`my-2 ${alignment}`}
        style={{
          minWidth:"200px", 
          maxWidth:"90%",
        }}
      >
        <div 
          className={'d-flex mb-2'}
        >
          <div className={'order-2 flex-fill px-3 py-2 mx-1'} 
            style={{
              minHeight:"60px", 
              backgroundColor:color,
              borderRadius:"15px",
              wordBreak:"break-word"
            }}
          >
            {"type" in props && props.type !== "typing" &&
              <p style={{fontVariant:"small-caps", size:".5em", marginBottom:"2px"}}>{props.type.toLowerCase()}</p>
            }
            <MessageContent {...props}/>
          </div>
          {icon}
        </div>
        {props.latest &&
          <QuickResponses {...props}/>
        }
      </div>
    );
  }
}

function QuickResponses(props){
  const responses = []

  function _changePage(){
    props.sendMessage("change-page", {})
  }

  function _rejectFunctions(){
    props.sendMessage("reject-functions", {functions:props.functions, text:props.functions})
  }

  function _rejectFunction(){
    props.sendMessage("reject-functions", {functions:[props.function], text:[props.function]})
  }

  function _eliSugg(){
    props.sendMessage("eli-sugg", {})
  }

  function _eliSuggAll(){
    props.sendMessage("eli-sugg-all", {})
  }

  function _dontKnow(){
    props.sendMessage("dont-know", {})
  }

  function _rejectKws(){
    props.sendMessage("reject-kws", {keywords:props.keywords, text:props.keywords})
  }

  switch(props.type) {
    case "sugg-all-s":
    case "change-page-s":
      if(props.functions && props.functions.length>0){
        responses.push({text: "Show me more", action:_changePage})
        responses.push({text: "None of these", action:_rejectFunctions})
        responses.push({text: "Show best function", action:_eliSugg})
      }
      break;

    case "info-s":
    case "sugg-s":
      if(props.function){
        responses.push({text: "Not this one", action:_rejectFunction})
        responses.push({text: "Next function", action:_eliSugg})
        responses.push({text: "List results", action:_eliSuggAll})
      }
      break;

    case "sugg-info-all-s":
    case "info-all-s":
      if(props.function){
        responses.push({text: "Not this one", action:_rejectFunction})
        responses.push({text: "Next function", action:_eliSugg})
        responses.push({text: "List results", action:_eliSuggAll})
      }
      break;

    case "eli-kw-s":
      responses.push({text: "Unsure", action:_dontKnow})
      responses.push({text: "None of these", action:_rejectKws})
      responses.push({text: "Show results", action:_eliSuggAll})
      responses.push({text: "Show best function", action:_eliSugg})
      break;

    case "eli-query-s":
      responses.push({text: "Not sure", action:_dontKnow})
      responses.push({text: "Show results", action:_eliSuggAll})
      responses.push({text: "Show best function", action:_eliSugg})
      break;
  }

  if(responses){
    return(
      <div style={{width:"100%", display:"flex", flexWrap:"wrap", justifyContent:"flex-end"}}>
        {responses.map((response,index)=>{return(
            <QuickResponse key={index} label={response.text} onClick={response.action}/>
          )
        })}
      </div>
    )
  }else{
    return(null)
  }
}

function MessageContent(props){
  var text = ""
  switch(props.type) {
    //User types

    case "eli-sugg":
      text += "Give me a suggestion."
      return <TextContent {...props} text={text}/>

    case "eli-sugg-all":
      text += "Show me all results."
      return <TextContent {...props} text={text}/>

    case "change-page":
      text = "Show me more results."
      return <TextContent {...props} text={text}/>


    case "eli-info":
      text = "Show me the " + props.feature + " documentation for " +  props["function"] + "."
      return <TextContent {...props} text={text}/>

    case "eli-info-all":
      text = "Show me all documentation for " +  props["function"] + "."
      return <TextContent {...props} text={text}/>


    case "reject-kws":
      if(props["keywords"].length==1){
        text = "I'm not interested in "+props["keywords"][0]+" as a keyword."
      }else{
        text = "I'm not interested in these as keywords: " +  props["keywords"].join(" ") + "."
      }
      return <TextContent {...props} text={text}/>

    case "reject-functions":
      text = "I'm not interested in these functions: " +  props["functions"].join(" ") + "."
      return <TextContent {...props} text={text}/>

    case "dont-know":
      text = "[...]"
      return <TextContent {...props} text={text}/>


    //System types
    case "typing":
      return <img style={{height:"50px"}} src={typingAnim}/>

    case "START":
      return <TextContent {...props}/>

    case "info-s":
    case "info-all-s":
      text = "Here you go:"
      return <InfoContent {...props} text={text}/>

    case "sugg-s":
      if(!props.function){
        text = "I'm sorry, I couldn't seem to find any functions to recommend. Please feel free to ask me about a specific function, or try a different search."
        return <TextContent {...props} text={text}/>
      }
      text = "I found this function. Would you like to know more about it?"
      return <SuggContent {...props} text={text}/>

    case "sugg-info-all-s":
      if(!props.function){
        text = "I'm sorry, I couldn't seem to find any functions to recommend. Please feel free to ask me about a specific function, or try a different search."
        return <TextContent {...props} text={text}/>
      }
      text = "Would the function " + props["function"] +" be helpful? Here's some information:"
      return <InfoContent {...props} text={text}/>


    case "sugg-all-s":
      if(!props.functions || props.functions.length==0){
        text = "I'm sorry, I couldn't seem to find any functions to list. Please feel free to ask me about a specific function, or try a different search."
        return <TextContent {...props} text={text}/>
      }
      text = "I found these functions. Would you like to know more about any of them?"
      return <ListContent {...props} text={text}/>

    case "change-page-s":
      if(!props.functions || props.functions.length==0){
        text = "I'm sorry, I couldn't seem to find any functions to list. Please feel free to ask me about a specific function, or try a different search."
        return <TextContent {...props} text={text}/>
      }
      text = "Here are some more results."
      return <ListContent {...props} text={text}/>


    case "eli-kw-s":
      text = "Do any of these keywords look relevant to your search?"
      return <KeywordContent {...props} text={text}/>

    case "eli-query-s":
      text = "Would you be able to tell me more about what you're looking for?"
      return <TextContent {...props} text={text}/>


    default:
      return <TextContent {...props}/>
  }
}

function TextContent(props){
  return(
    <div style={{whiteSpace:"pre-line"}}>
      {props.text?props.text:""}
    </div>
  )
}

function ListContent(props){

  const functions = useContext(FunctionsContext)

  // const _handleClick = (func) =>{
  //   props.setInput("@"+func)
  // }
  
  const classes = useStyles();

  
  return(
    <div>
      {props.text}
      <hr/>
      {props.functions.map((func)=>{
        return(
          <div
            onClick={(e)=>(props.setInput("@"+func))}
            className={classes.link}
          > 
            <b>> {func}</b>
          </div>
        )
      })}
    </div>
  )
}

function SuggContent(props){

  const functions = useContext(FunctionsContext)
  const [expanded, setExpanded] = useState("")
  const classes = useStyles();

  const _handleChange = (panel) => (event, isExpanded) => {
    console.log(event.target)
    if(isExpanded) event.target.scrollIntoView({behavior: "smooth", block: "nearest", inline: "nearest"})
    setExpanded(isExpanded ? panel : false);
  };

  const _handleButton = (func, feature) => (event) =>{
    if(feature === "all"){
      props.sendMessage("eli-info-all", {"function":func})
    }else{
      props.sendMessage("eli-info", {"function":func, "feature":feature})
    }
  }

  return(
    <div>
      {props.text}
      <div
        onClick={(e)=>(props.setInput("@"+props.function))}
        className={classes.link}
      > 
        <b>> {props.function}</b>
      </div>
      <hr/>
      {functions[props.function]["Summary"]}
    </div>
  )
}

function KeywordContent(props){
  const _handleButton = (keywords) => (event) =>{
    props.sendMessage("provide-kw", {"keyword":keywords, "text":keywords})
  }

  return(
    <div>
      <p> {props.text}</p>
      <InputWrapper>
        {props.keywords.map((kw, i)=>{
          return(
            <MTag key={i} label={kw} onClick={(e)=>(props.setInput("+"+kw))}/>
          )
        })}
      </InputWrapper>
    </div>
  )
}

function InfoContent(props){
  const functions = useContext(FunctionsContext)
  const func = props.function
  const classes = useStyles();
  if(props.type=="info-s"){
    return(
      <div>
          <p> {props.text}</p>
          {/*<h3 
            className={classes.link}
            onClick={(e)=>(props.setInput("@"+func))}
          >
            {func}
          </h3>*/}
          <div
            onClick={(e)=>(props.setInput("@"+func))}
            className={classes.link}
          > 
            <b>> {func}</b>
          </div>
          <hr/>
          <b>{props.feature}</b>
          <p> {functions[func][props.feature]} </p>
      </div>
    )
  }else{
    //It's info-all
    return(
      <div>
        <p> {props.text}</p>
        <div 
            className={classes.link}
            onClick={(e)=>(props.setInput("@"+func))}
          >
          <b>> {func}</b>
        </div>
        <hr/>
        {Object.keys(functions[func]).map(key=>{
          if(!["All", "Summary"].includes(key)){
            return(
              <div key={key}> 
                <b>{key}</b>
                <p>{functions[func][key]}</p>
              </div>
            )
          }
        })}
      </div>
    )
  }
}

const useStyles = makeStyles((theme) => ({
  root: {
    padding: '2px 4px',
    display: 'flex',
    alignItems: 'center',
    width: "100%",
  },
  input: {
    marginLeft: theme.spacing(1),
    flex: 1,
  },
  iconButton: {
    padding: 10,
  },
  divider: {
    height: 28,
    margin: 4,
  },
  link:{
    color:"#001081",
    cursor:"pointer",
    "&:hover":{
      textDecoration:"underline"
    }
  }
}));



function Searchbar(props){

  const [queryColor, setQueryColor] = useState("black");
  const [resources, setResources] = useState([]);
  const functions = useContext(FunctionsContext)
  const [functionList, setFunctionList] = useState(new Set())


  useEffect(() => {
    setFunctionList(new Set(Object.keys(functions)))
  }, [functions]);

  useEffect(() => {
    if(isFunction()){
       setQueryColor("blue")
       setFunctionResources(isFunction())
    }else{
      if(resources){
        setResources([])
      }    
      if(isKeyword()){
       setQueryColor("green")
      }else{
         setQueryColor("black")
      }    
    }
  }, [props.input]);

  function _submit(){
    if(isKeyword()){
      _addKeyword(isKeyword())
    }else if(isFunction()){
      _eliInfo("All", isFunction())
    }else{
      _submitQuery()
    }
    setQueryColor("black")
    props.setInput("")
  }

  function _submitQuery(){
    if(props.input.trim().length > 0){
      props.sendMessage("provide-query", {query:props.input.trim(), text:props.input.trim()})
    }else{
      props.sendMessage("eli-sugg")
    }
  }

  function _addKeyword(keyword){
    if(!props.keywords.includes(keyword)){
      props.setKeywords([...props.keywords, keyword])
      props.sendMessage("provide-kw", {keyword:keyword, text:keyword})
    }
  }

  function _eliInfo(feature, func=null){
    if(!func){
      func = isFunction()
    }
    if(feature=="All"){
      props.sendMessage("eli-info-all", {"function":func})
    }else{
      props.sendMessage("eli-info", {"function":func, feature:feature})
    }
  }

  function isKeyword(){
    const regex = /^\+(\w*)\s*$/;
    if(props.input.match(regex)){
      return props.input.match(regex)[1]
    }
    return null
  }

  function isFunction(){
    const regex = /^@(\w*)\s*$/;
    if(props.input.match(regex)){
      const name = props.input.match(regex)[1]
      if(functionList.has(name)){
        return name
      }
    }
    return null
  }

  function _onDelete(keyword){
    if(props.keywords.includes(keyword)){
      props.setKeywords(props.keywords.filter(kw=>kw!=keyword))
      props.sendMessage("reject-kws", {keywords:[keyword], text:keyword})
    }
  }

  function _handleInputChange(e){
    if(e){
      props.setInput(e.target.value);
    }
  }

  function setFunctionResources(func){
    if(props.taskset=="libssh"){
      const _resources = Object.keys(functions[func]).filter(feat=>!["All", "Summary"].includes(feat))
      setResources(_resources)
    }else{
      const _resources = Object.keys(functions[func]).filter(feat=>!["All", "Summary", "Returns", "Parameters"].includes(feat))
      setResources(_resources)
    }
  }

  const classes = useStyles();

  return(
    <div className="w-100 px-2 py-1 " style={{backgroundColor:"#f2f8ff"}}>
      <InputWrapper>
        {resources.map((option, index) => (
          <DocTag key={index} label={option} onClick={(e)=>(_eliInfo(option))}/>
        ))}
      </InputWrapper>
      <div className=" py-2">

        <Paper component="form" style={{width:"100%"}}>
          <InputWrapper>
            {props.keywords.map((option, index) => (
              <Tag key={index} label={option} onDelete={(e)=>(_onDelete(option))}/>
            ))}
          </InputWrapper>
          <div className={classes.root}>
            <InputBase
              className={classes.input}
              style={{color:queryColor}}
              value = {props.input}
              autoComplete="off"
              inputProps={{ 'aria-label': 'search',}}
              placeholder="Search..."
              onChange={_handleInputChange}
              onKeyPress={(evt)=>{
                if(evt.key === 'Enter'){
                  evt.preventDefault();
                  _submit()
                }
              }}
            />
            {props.input &&
              <Tooltip title="Cancel">
                <IconButton className="nofocus" size="medium" onClick={()=>(props.setInput(""))}>
                  <CancelIcon style={{fill: "#c8c8c8"}} fontSize="inherit"/>
                </IconButton>
              </Tooltip>
            }
            <div >
            <Tooltip  title="Submit">
              <IconButton className="nofocus" size="medium" onClick={_submit}>
                <SearchIcon fontSize="inherit"/>
              </IconButton>
            </Tooltip>
            </div>
          </div>
        </Paper>
      </div>
    </div>
  )
}
// className="border-start ms-1 ps-1"

const Label = styled("label")`
  padding: 0 0 4px;
  line-height: 1.5;
  display: block;
`;

// border: 1px solid #d9d9d9;
//   background-color: #fff;
//   border-radius: 4px;

const InputWrapper = styled("div")`
  width: 300px;
  padding: 1px;
  display: flex;
  flex-wrap: wrap;
`;

const DocTag = styled(({ label, ...props }) => (
  <div {...props} style={{backgroundColor: randomColor({luminosity:"light", seed:label})}}>
  

    <span>{label}</span>
  </div>
))`
  display: flex;
  align-items: center;
  height: 24px;
  margin: 2px;
  line-height: 22px;
  border: 1px solid #e8e8e8;
  border-radius: 20px;
  box-sizing: content-box;
  padding: 4px 6px;
  outline: 0;
  cursor: pointer;
  opacity: 1;

  &:focus {
    border-color: #40a9ff;
  }

  &:hover{
    opacity: .6;
  }

  & span {
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }

  & svg {
    font-size: 12px;
    cursor: pointer;
    padding: 4px;
  }
`;

const Tag = styled(({ label, onDelete, ...props }) => (
  <div {...props}>
    <span>{label}</span>
    <CloseIcon onClick={onDelete} fontSize="large"/>
  </div>
))`
  display: flex;
  align-items: center;
  height: 24px;
  margin: 2px;
  line-height: 22px;
  background-color: #fafafa;
  border: 1px solid #e8e8e8;
  border-radius: 2px;
  box-sizing: content-box;
  padding: 0 4px 0 10px;
  outline: 0;
  overflow: hidden;

  &:focus {
    border-color: #40a9ff;
    background-color: #e6f7ff;
  }

  & span {
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }

  & svg {
    font-size: 18px;
    cursor: pointer;
    padding: 4px;
  }
`;

const MTag = styled(({ label, onClick, ...props }) => (
  <div {...props} onClick={onClick}>
    <span>{label}</span>
  </div>
))`
  display: flex;
  align-items: center;
  height: 24px;
  margin: 2px;
  line-height: 22px;
  background-color: #fafafa;
  border: 1px solid #e8e8e8;
  border-radius: 2px;
  box-sizing: content-box;
  padding: 4px 6px;
  outline: 0;
  opacity: 1;
  cursor: pointer;

  &:focus {
    border-color: #40a9ff;
  }

  &:hover{
    opacity: .5;
  }

  & span {
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }

  & svg {
    font-size: 18px;
    cursor: pointer;
    padding: 4px;
  }
`;

const QuickResponse = styled(({label, onClick, ...props}) => (
  <div {...props} onClick={onClick}>
    <span>{label}</span>
  </div>
))`
  display: flex;
  align-items: center;
  height: 24px;
  margin: 2px;
  line-height: 22px;
  background-color: #38FaD5;
  border: 1px solid #e8e8e8;
  border-radius: 20px;
  box-sizing: content-box;
  padding: 4px 6px;
  outline: 0;
  opacity: 1;
  cursor: pointer;

  &:focus {
    border-color: #40a9ff;
  }

  &:hover{
    opacity: .5;
  }

  & span {
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }

  & svg {
    font-size: 18px;
    cursor: pointer;
    padding: 4px;
  }
`;


export default Chatbox