import axios from 'axios';
// ActionProvider starter code
class ActionProvider {
  constructor(createChatBotMessage, setStateFunc, createClientMessage) {
    this.createChatBotMessage = createChatBotMessage;
    this.setState = setStateFunc;
    this.createClientMessage = createClientMessage;
  }

  // 1. Update language state. Triggered when users select what language they'd prefer at the beginning.
  handleLanguageButtons = (userID, sessionID, userInput, userInputType) => {
    // userInput = language selected
    let language = userInput

    // Output user's selection as a message in the chatbot
    let message = this.createClientMessage(userInput);  // creates a full dict with id etc.
    this.addMessageToBotState(message);                 

    // Update language state based on what users selected
    this.setState((state) => ({
      ...state,
      language: userInput,
    }))

    // Proceed to ask users for their username
    return this.askForUsername(language)
  };

  // 2. Ask users for username. Triggered after users selected their language of choice.
  askForUsername = (language) => {
    // The askForUsername prompt in different languages
    let m = {
      "English(EN)": "Please enter your username:",
      "中文(ZH)":  "请输入您的用户名:",
    }

    const messages = this.createChatBotMessage(
      m[language],
      {
        withAvatar: true,
      }
    );

    this.addMessageToBotState(messages);

    // Does not return anything.
    // Wait for users to respond with username. Reply processed by updateUsername triggered by MessageParser.js
  }

  // 3. Update the username state. Triggered after users input their username.
  updateUsername = (username, language) => {
    this.setState((state) => ({
      ...state,
      username: username,
    }));

    // Proceed to ask users for their password
    return this.askForPassword (language)
  }

  // 4. Ask for password. Triggered after username state updated.
  askForPassword = (language) => {
    // The askForPassword prompt in different languages
    let m = {
      "English(EN)": "Please enter your password:",
      "中文(ZH)": "请输入您的用户密码:",
    }

    // Output the message
    const messages = this.createChatBotMessage(
      m[language],
      {
        withAvatar: true,
      }
    );

    this.addMessageToBotState(messages);
    
    // Does not return anything.
    // Wait for users to respond with password. Reply processed by updateUserID triggered by MessageParser.js
  }

  // Checking for ID with a request
  updateUserID = async (username, password, language) => {
    // Update the password state
    this.setState((state) => ({
      ...state,
      password: password,
    }));

    // Send to backend to verify validity.
    // URL to use for AWS (Axios requests)
    // const uri = `/api/login`

    // URL to use for local requests
    const uri = `http://localhost:5000/api/login`
    let user_info = {
      username: username,
      password: password,
      language: language,
    };

    const response = await axios.post(uri, {
      user_info
    })

    // dataReceived from backend. 
    // format: response = {data: {validID : bool, userID: string}}
    let dataReceived = response.data

    // invalid match:
    if (!dataReceived.validID) {
      let m = {
        "English(EN)": "The user ID and password combination is not valid. Please enter user ID again.",
        "中文(ZH)": "对不起，您所输入的用户名或密码不正确。请重新输入您的用户名。"
      };

      // Output error message
      let message = this.createChatBotMessage(
        m[language],
        {withAvatar: true}
      );

      this.addMessageToBotState(message);
      
      // Clear the username and password state
      this.setState((state) => ({
        ...state,
        username: null,
        password: null
      }));
    } 
    
    // valid match:
    else {
      let m = {
        "English(EN)": "Login successful! ",
        "中文(ZH)": "登录成功！",
      }

      let model_prompt = dataReceived.model_prompt
      this.setState((state) => ({ ...state, userState: dataReceived.userID, inputType: dataReceived.choices, sessionID: dataReceived.sessionID }));
      let message = this.createChatBotMessage(
        m[language], 
        {withAvatar: true}
      );

      // Opening prompt -> open text
      this.addMessageToBotState(message);
      message = this.createChatBotMessage(
        model_prompt, 
        {withAvatar: true}
      );
      this.addMessageToBotState(message);
    }
  };

  // Send API request
  sendRequest = async (choice_info) => {
    // URL to use for AWS (Axios requests)
    // const uri = `/api/update_session`

    // URL to use for local requests
    const uri = `http://localhost:5000/api/update_session`;
    const response = await axios.post(uri, {
      choice_info
    })

    this.handleReceivedData(response.data);
  };

  handleReceivedData = (dataReceived) => {
    // dataReceived = {
    //   chatbot_response: "This is the chatbot message to display",
    //   user_options: options to map to expected buttons below
    // }

    const userOptions = dataReceived.user_options
    let optionsToShow = null;


    //  Required options: null or "YesNo" or "Continue" or "Feedback" or "Emotion"}
    if (userOptions.length === 1 && (userOptions[0] === "open_text" || userOptions[0] === "any")) {
      optionsToShow = null;
    } else if (userOptions.length === 1 && userOptions[0] === "continue") {
      optionsToShow = "Continue"
    } else if (userOptions.length === 2 && userOptions[0] === "yes" && userOptions[1] === "no") {
      optionsToShow = "YesNo"
    } else if (userOptions.length === 2 && userOptions[0] === "yes, i would like to try one of these protocols" && userOptions[1] === "no, i would like to try something else") {
      optionsToShow = "YesNoProtocols"
    } else if (userOptions.length === 2 && userOptions[0] === "recent" && userOptions[1] === "distant") {
      optionsToShow = "RecentDistant"
    } else if (userOptions.length === 3 && userOptions[0] === "positive" && userOptions[1] === "neutral" && userOptions[2] === "negative") {
      optionsToShow = "Emotion"
    } else if (userOptions.length === 3 && userOptions[0] === "better" && userOptions[1] === "worse" && userOptions[2] === "no change") {
      optionsToShow = "Feedback"
    } else {
      // Protocol case
      optionsToShow = "Protocol"
      this.setState((state) => ({
        ...state,
        protocols: userOptions,
        askingForProtocol: true
      }));
    }
    this.setState((state) => ({
      ...state,
      currentOptionToShow: optionsToShow,
    }));

    // Display chatbot message
    // (i) Only a single sentence
    if (typeof dataReceived.chatbot_response === "string") {
      const messages = this.createChatBotMessage(dataReceived.chatbot_response, {
        withAvatar: true,
        widget: optionsToShow,
      });
      this.addMessageToBotState(messages);
    } 
    // (ii) Multiple sentence bubbles 
    else {
      for (let i = 0; i < dataReceived.chatbot_response.length; i++) {
        let widget = null;
        // Shows options after last message
        if (i === dataReceived.chatbot_response.length - 1) {
          widget = optionsToShow;
        }
        const message_to_add = this.createChatBotMessage(dataReceived.chatbot_response[i], {
          withAvatar: true,
          widget: widget,
        });
        this.addMessageToBotState(message_to_add);
      }

    }
  };

  handleButtonsEmotion = (userID, sessionID, userInput, userInputType) => {
    let inputToSend = userInput;
    let message = this.createClientMessage(userInput);
    this.addMessageToBotState(message);


    // Ignores input type above and manually defines; other cases will need an if check for this
    let input_type = ["positive", "neutral", "negative"]
    const dataToSend = {
      user_id: userID,
      session_id: sessionID,
      user_choice: inputToSend,
      input_type: input_type,
    };
    this.sendRequest(dataToSend);
  }

  handleButtons = (userID, sessionID, userInput, userInputType) => {
    let message = this.createClientMessage(userInput);
    this.addMessageToBotState(message);

    const dataToSend = {
      user_id: userID,
      session_id: sessionID,
      user_choice: userInput,
      input_type: userInputType,
    };
    return this.sendRequest(dataToSend);
  };

  askForProtocol = () => {
    let message = "Please type a protocol number (1-20), using the workshops to help you."
    this.addMessageToBotState(message);
    this.setState((state) => ({
      ...state,
      askingForProtocol: true,
    }))
  }

  stopAskingForProtocol = () => {
    this.setState((state) => ({
      ...state,
      askingForProtocol: false,
    }))
  }


  // Copies last message from model
  copyLastMessage = () => {
    this.setState((state) => ({
      ...state,
      messages: [...state.messages, state.messages[state.messages.length - 2]],
    }))
  }


  // Add message to state
  addMessageToBotState = (message) => {
    this.setState((state) => ({
      ...state,
      messages: [...state.messages, message],
    }));
  };
}

export default ActionProvider;
