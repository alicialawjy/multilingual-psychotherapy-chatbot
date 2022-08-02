// Config starter code
import React from "react";
import { createChatBotMessage } from "react-chatbot-kit";

import LanguageOptions from "./widgets/options/GeneralOptions/LanguageOptions";
import YesNoOptions from "./widgets/options/GeneralOptions/YesNoOptions";
import ProtocolOptions from "./widgets/options/GeneralOptions/ProtocolOptions";
import ContinueOptions from "./widgets/options/GeneralOptions/ContinueOptions";
import FeedbackOptions from "./widgets/options/GeneralOptions/FeedbackOptions";
import EmotionOptions from "./widgets/options/GeneralOptions/EmotionOptions";
import EventOptions from "./widgets/options/GeneralOptions/EventOptions";
import YesNoProtocolOptions from "./widgets/options/GeneralOptions/YesNoProtocolsOptions";
const botName = "SATbot";

const config = {
  botName: botName,
  initialMessages: [
    // For ease of reading, write in 2 separate sentences
    createChatBotMessage("Hello! Welcome to SATbot. Please select your language of choice for today's session!", 
    {
      withAvatar: true,
    }),
    createChatBotMessage("你好，欢迎来到SATbot。请问您想用以下哪个语言来进行今天的活动？",
    {
      withAvatar: true,
      widget: "Language", // show widget only at the end sentence
    }),
  ],

  state: {
    userState: null,
    username: null,
    password: null,
    language: null,
    sessionID: null,
    protocols: [],
    askingForProtocol: false
  },

  customComponents: {
    header: () => <div class = "chatbot-header">S A T b o t</div>,
    botAvatar: () => <div class="react-chatbot-kit-chat-bot-avatar-container"><p class="react-chatbot-kit-chat-bot-avatar-letter">S</p></div>
  },

  widgets: [
    {
      widgetName: "Language",
      widgetFunc: (props) => <LanguageOptions {...props} />,
      mapStateToProps: ["userState", "sessionID"],
    },
    {
      widgetName: "YesNo",
      widgetFunc: (props) => <YesNoOptions {...props} />,
      mapStateToProps: ["userState", "sessionID"],
    },
    {
      widgetName: "Continue",
      widgetFunc: (props) => <ContinueOptions {...props} />,
      mapStateToProps: ["userState", "sessionID"],
    },
    {
      widgetName: "Emotion",
      widgetFunc: (props) => <EmotionOptions {...props} />,
      mapStateToProps: ["userState", "sessionID"],
    },
    {
      widgetName: "Feedback",
      widgetFunc: (props) => <FeedbackOptions {...props} />,
      mapStateToProps: ["userState", "sessionID"],
    },
    {
      widgetName: "Protocol",
      widgetFunc: (props) => <ProtocolOptions {...props} />,
      mapStateToProps: ["userState", "sessionID", "protocols", "askingForProtocol"],
    },
    {
      widgetName: "YesNoProtocols",
      widgetFunc: (props) => <YesNoProtocolOptions {...props} />,
      mapStateToProps: ["userState", "sessionID"],
    },
    {
      widgetName: "RecentDistant",
      widgetFunc: (props) => <EventOptions {...props} />,
      mapStateToProps: ["userState", "sessionID"],
    },
  ],
};

export default config;
