import React from "react";
import { Chatbot } from "react-chatbot-kit";
import "./App.css";
import MessageParser from "./MessageParser";
import ActionProvider from "./ActionProvider";
import config from "./config";
// import MoreInfoDocs from "./widgets/docs/MoreInfoDocs";

function App() {
  return (
    <div className="App">
      <header className="app-chatbot-container">
        <Chatbot
          config={config}
          messageParser={MessageParser}
          actionProvider={ActionProvider}
        />
      </header>
      {/* <MoreInfoDocs className="more-info" /> */}
    </div>
  );
}

export default App;
