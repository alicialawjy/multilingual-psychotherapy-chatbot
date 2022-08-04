// MessageParser starter code
class MessageParser {
  constructor(actionProvider, state) {
    this.actionProvider = actionProvider;
    this.state = state;
  }

  // This method is called inside the chatbot when it receives a message from the user.
  parse(message) {
    // Message received was the username. Update it.
    if (this.state.username == null) {
      return this.actionProvider.updateUsername(message, this.state.language);
    } 
    
    // Message received was the password. Update and verify.
    else if (this.state.password == null) {
      return this.actionProvider.updateUserID(this.state.username, message, this.state.language);
    } 

    
    // Protocols
    else if (this.state.askingForProtocol && parseInt(message) >= 1 && parseInt(message) <= 20) {

      const choice_info = {
        user_id: this.state.userState,
        session_id: this.state.sessionID,
        user_choice: message,
        input_type: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      };
      this.actionProvider.stopAskingForProtocol()

      return this.actionProvider.sendRequest(choice_info);
    } 

    // Asking for protocols beyond 1-20
    else if (this.state.askingForProtocol && (parseInt(message) < 1 || parseInt(message) > 20)) {
      return this.actionProvider.askForProtocol()
    }

    else {
      let input_type = null;
      if (this.state.inputType.length === 1) {
        input_type = this.state.inputType[0]
      } else {
        input_type = this.state.inputType
      }
      const currentOptionToShow = this.state.currentOptionToShow

      // if user types instead of selecting an option, redo the prompt:
      if ((currentOptionToShow === "Continue" && message !== "Continue") ||
        (currentOptionToShow === "Emotion" && (message !== "Happy" && message !== "Sad" && message !== "Angry" && message !== "Neutral")) ||
        (currentOptionToShow === "RecentDistant" && (message !== "Recent" && message !== "Distant")) ||
        (currentOptionToShow === "Feedback" && (message !== "Better" && message !== "Worse" && message !== "No change")) ||
        (currentOptionToShow === "Protocol" && (!this.state.protocols.includes(message))) ||
        (currentOptionToShow === "YesNo" && (message !== "Yes" && message !== "No"))
      ) {
        this.actionProvider.copyLastMessage()
      } 
      
      // otherwise, send the request to backend for processing
      // accessed only when asked "how are you feeling"
      else {
        const choice_info = {
          user_id: this.state.userState,      // userID [int]
          session_id: this.state.sessionID,   // sessionID [int]
          user_choice: message,               // the parsed message [str] 
          input_type: input_type,             // input type [str] = open_text (in this case)
        };

        return this.actionProvider.sendRequest(choice_info);
      }
    }
  }
}

export default MessageParser;
