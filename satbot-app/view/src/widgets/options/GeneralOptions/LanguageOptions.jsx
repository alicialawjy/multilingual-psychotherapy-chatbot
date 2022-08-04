import React from "react";
import Options from "../Options/Options";

const LanguageOptions = (props) => {
  const options = [
    {
      name: "English(EN)",
      handler: props.actionProvider.handleLanguageButtons,
      id: 16,
      userID: props.userState,
      sessionID: props.sessionID,
      userInputType: "Language",
    },
    {
      name: "中文(ZH)",
      handler: props.actionProvider.handleLanguageButtons,
      id: 17,
      userID: props.userState,
      sessionID: props.sessionID,
      userInputType: "Language",
    },
  ];

  return <Options options={options} />;
};
export default LanguageOptions;
