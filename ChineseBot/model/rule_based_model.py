from lib2to3.pgen2.pgen import DFAState
from locale import D_FMT
from tkinter import E
import nltk
from regex import P
from model.models import UserModelSession, Choice, UserModelRun, Protocol
from model.classifiers import get_emotion
import pandas as pd
import numpy as np
import random
from collections import deque
import re
import datetime
import time


class ModelDecisionMaker:
    def __init__(self):
        # Titles from workshops (Title 7 adapted to give more information)
        self.PROTOCOL_TITLES = [
            "0: None",
            "1: Connecting with the Child [Week 1]",
            "2: Laughing at our Two Childhood Pictures [Week 1]",
            "3: Falling in Love with the Child [Week 2]",
            "4: Vow to Adopt the Child as Your Own Child [Week 2]",
            "5: Maintaining a Loving Relationship with the Child [Week 3]",
            "6: An exercise to Process the Painful Childhood Events [Week 3]",
            "7: Protocols for Creating Zest for Life [Week 4]",
            "8: Loosening Facial and Body Muscles [Week 4]",
            "9: Protocols for Attachment and Love of Nature  [Week 4]",
            "10: Laughing at, and with One's Self [Week 5]",
            "11: Processing Current Negative Emotions [Week 5]",
            "12: Continuous Laughter [Week 6]",
            "13: Changing Our Perspective for Getting Over Negative Emotions [Week 6]",  # noqa
            "14: Protocols for Socializing the Child [Week 6]",
            "15: Recognising and Controlling Narcissism and the Internal Persecutor [Week 7]",  # noqa
            "16: Creating an Optimal Inner Model [Week 7]",
            "17: Solving Personal Crises [Week 7]",
            "18: Laughing at the Harmless Contradiction of Deep-Rooted Beliefs/Laughing at Trauma [Week 8]",  # noqa
            "19: Changing Ideological Frameworks for Creativity [Week 8]",
            "20: Affirmations [Week 8]",
        ]

        self.TITLE_TO_PROTOCOL = {
            self.PROTOCOL_TITLES[i]: i for i in range(len(self.PROTOCOL_TITLES))
        }

        self.recent_protocols = deque(maxlen=20)
        self.reordered_protocol_questions = {}
        self.protocols_to_suggest = []

        # Goes from user id to actual value
        self.current_run_ids = {}
        self.current_protocol_ids = {}

        self.current_protocols = {}

        self.positive_protocols = [i for i in range(1, 21)]

        self.INTERNAL_PERSECUTOR_PROTOCOLS = [
            self.PROTOCOL_TITLES[15],
            self.PROTOCOL_TITLES[16],
            self.PROTOCOL_TITLES[8],
            self.PROTOCOL_TITLES[19],
        ]

        # Keys: user ids, values: dictionaries describing each choice (in list)
        # and current choice
        self.user_choices = {}

        # Keys: user ids, values: scores for each question
        #self.user_scores = {}

        # Keys: user ids, values: current suggested protocols
        self.suggestions = {}

        # Tracks current emotion of each user after they classify it
        self.user_emotions = {}

        self.guess_emotion_predictions = {}
        # Structure of dictionary: {question: {
        #                           model_prompt: str or list[str],
        #                           choices: {maps user response to next protocol},
        #                           protocols: {maps user response to protocols to suggest},
        #                           }, ...
        #                           }
        # This could be adapted to be part of a JSON file (would need to address
        # mapping callable functions over for parsing).

        self.users_names = {}
        self.remaining_choices = {}

        # self.recent_questions = {}

        self.language = {}
        self.datasets = {}

        self.QUESTIONS = {
            "ask_feeling": {
                "model_prompt": lambda user_id: self.ask_feeling(user_id), #db_session, curr_session, app
                "choices": {
                    "open_text": lambda user_id: self.determine_next_prompt_opening(user_id)
                },
                "protocols": {"open_text": []},
            },

            "guess_emotion": {
                "model_prompt": lambda user_id: self.get_model_prompt_guess_emotion(user_id),
                "choices": {
                    "yes": {
                        "Sad": "after_classification_negative",
                        "Angry": "after_classification_negative",
                        "Anxious/Scared": "after_classification_negative",
                        "Happy/Content": "after_classification_positive",
                    },
                    "no": "check_emotion",
                },
                "protocols": {
                    "yes": [],
                    "no": []
                    },
            },

            "check_emotion": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name = "All emotions - I am sorry. Please select from the emotions below the one that best reflects what you are feeling:"
                    ),

                "choices": {
                    "Sad": lambda user_id: self.get_after_classification(user_id, emotion='Sad'),
                    "Angry": lambda user_id: self.get_after_classification(user_id, emotion='Angry'),
                    "Anxious/Scared": lambda user_id: self.get_after_classification(user_id, emotion='Anxious/Scared'),
                    "Happy/Content": lambda user_id: self.get_after_classification(user_id, emotion="Happy/Content"),
                },
                "protocols": {
                    "Sad": [],
                    "Angry": [],
                    "Anxious/Scared" : [],
                    "Happy/Content": []
                },
            },

            ############# NEGATIVE EMOTIONS (SADNESS, ANGER, FEAR/ANXIETY)
            "after_classification_negative": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name=" - Was this caused by a specific event/s?",
                    special=True
                    ),
                "choices": {
                    "Yes, something happened": "event_is_recent",
                    "No, it's just a general feeling": "more_questions",
                },
                "protocols": {
                    "Yes, something happened": [],
                    "No, it's just a general feeling": []
                },
            },

            "event_is_recent": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name=" - Was this caused by a recent or distant event (or events)?",
                    special=True
                    ),

                "choices": {
                    "It was recent": "revisiting_recent_events",
                    "It was distant": "revisiting_distant_events",
                },
                "protocols": {
                    "It was recent": [],
                    "It was distant": []
                    },
            },

            "revisiting_recent_events": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name=" - Have you recently attempted protocol 11 and found this reignited unmanageable emotions as a result of old events?",
                    special=True
                    ),
                "choices": {
                    "yes": "more_questions",
                    "no": "more_questions",
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[7], self.PROTOCOL_TITLES[8]],
                    "no": [self.PROTOCOL_TITLES[11]],
                },
            },

            "revisiting_distant_events": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name=" - Have you recently attempted protocol 6 and found this reignited unmanageable emotions as a result of old events?",
                    special=True
                    ),

                "choices": {
                    "yes": "more_questions",
                    "no": "more_questions",
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[17]],
                    "no": [self.PROTOCOL_TITLES[6]]
                },
            },

            "more_questions": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id,
                    column_name=" - Thank you. Now I will ask some questions to understand your situation.",
                    special=True
                    ),

                "choices": {
                    "Okay": lambda user_id: self.get_next_question(user_id),
                    "I'd rather not": "project_emotion",
                },
                "protocols": {
                    "Okay": [],
                    "I'd rather not": [self.PROTOCOL_TITLES[13]],
                },
            },

            "displaying_antisocial_behaviour": {
                "model_prompt": lambda user_id : self.get_model_prompt_antisocial(user_id),

                "choices": {
                    "yes": "project_emotion",
                    "no": lambda user_id: self.get_next_question(user_id),
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[14]],
                    "no": [self.PROTOCOL_TITLES[13]],
                },
            },

            "internal_persecutor_saviour": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name=" - Do you believe that you should be the saviour of someone else?",
                    special=True
                    ),
                "choices": {
                    "yes": "project_emotion",
                    "no": "internal_persecutor_victim",
                },
                "protocols": {
                    "yes": self.INTERNAL_PERSECUTOR_PROTOCOLS,
                    "no": []
                },
            },

            "internal_persecutor_victim": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name=" - Do you see yourself as the victim, blaming someone else for how negative you feel?",
                    special=True
                    ),
                "choices": {
                    "yes": "project_emotion",
                    "no": "internal_persecutor_controlling",
                },
                "protocols": {
                    "yes": self.INTERNAL_PERSECUTOR_PROTOCOLS,
                    "no": []
                },
            },

            "internal_persecutor_controlling": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name=" - Do you feel that you are trying to control someone?",
                    special=True
                    ),

                "choices": {
                    "yes": "project_emotion",
                    "no": "internal_persecutor_accusing"
                },
                "protocols": {
                    "yes": self.INTERNAL_PERSECUTOR_PROTOCOLS,
                    "no": []
                },
            },

            "internal_persecutor_accusing": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name=" - Are you always blaming and accusing yourself for when something goes wrong?",
                    special=True
                    ),
                "choices": {
                    "yes": "project_emotion",
                    "no": lambda user_id: self.get_next_question(user_id),
                },
                "protocols": {
                    "yes": self.INTERNAL_PERSECUTOR_PROTOCOLS,
                    "no": [self.PROTOCOL_TITLES[13]],
                },
            },

            "rigid_thought": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id,
                    column_name=" - In previous conversations, have you considered other viewpoints presented?",
                    special=True
                    ),

                "choices": {
                    "yes": lambda user_id: self.get_next_question(user_id),
                    "no": "project_emotion",
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13]],
                    "no": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[19]],
                },
            },


            "personal_crisis": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name=" - Are you undergoing a personal crisis (experiencing difficulties with loved ones e.g. falling out with friends)?",
                    special=True
                    ),

                "choices": {
                    "yes": "project_emotion",
                    "no": lambda user_id: self.get_next_question(user_id),
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[17]],
                    "no": [self.PROTOCOL_TITLES[13]],
                },
            },

            ################# POSITIVE EMOTION (HAPPINESS/CONTENT) #################

            "after_classification_positive": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name="Happy - That's Good! Let me recommend a protocol you can attempt."
                    ),

                "choices": {
                    "Okay": "suggestions",
                    "No, thank you": "ending_prompt"
                },
                "protocols": {
                    "Okay": [self.PROTOCOL_TITLES[9], self.PROTOCOL_TITLES[10], self.PROTOCOL_TITLES[11]], 
                    "No, thank you": []
                },
            },

            ############################# ALL EMOTIONS #############################

            "project_emotion": {
               "model_prompt": lambda user_id: self.get_model_prompt_project_emotion(user_id),

               "choices": {
                   "Continue": "suggestions",
               },
               "protocols": {
                   "Continue": [],
               },
            },


            "suggestions": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name= "All emotions - Here are my recommendations, please select the protocol that you would like to attempt"
                    ),

                "choices": {
                     self.PROTOCOL_TITLES[k]: "trying_protocol" for k in self.positive_protocols
                },
                "protocols": {
                     self.PROTOCOL_TITLES[k]: [self.PROTOCOL_TITLES[k]]
                     for k in self.positive_protocols
                },
            },

            "trying_protocol": {
                "model_prompt": lambda user_id: self.get_model_prompt_trying_protocol(user_id),

                "choices": {
                    "continue": "user_found_useful"
                },
                "protocols": {"continue": []},
            },

            "user_found_useful": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name = "All emotions - Do you feel better or worse after having taken this protocol?"
                    ),

                "choices": {
                    "I feel better": "new_protocol_better",
                    "I feel worse": "new_protocol_worse"
                },
                "protocols": {
                    "I feel better": [],
                    "I feel worse": []
                },
            },

            "new_protocol_better": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name="All emotions - Would you like to attempt another protocol? (Patient feels better)"
                    ),

                "choices": {
                    "Yes (show follow-up suggestions)": lambda user_id: self.determine_next_prompt_new_protocol(user_id),
                    "Yes (restart questions)": "restart_prompt",
                    "No (end session)": "ending_prompt",
                },
                "protocols": {
                    "Yes (show follow-up suggestions)": [],
                    "Yes (restart questions)": [],
                    "No (end session)": []
                },
            },

            "new_protocol_worse": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name="All emotions - Would you like to attempt another protocol? (Patient feels worse)"
                    ),

                "choices": {
                    "Yes (show follow-up suggestions)": lambda user_id: self.determine_next_prompt_new_protocol(user_id),
                    "Yes (restart questions)": "restart_prompt",
                    "No (end session)": "ending_prompt",
                },
                "protocols": {
                    "Yes (show follow-up suggestions)": [],
                    "Yes (restart questions)": [],
                    "No (end session)": []
                },
            },

            "ending_prompt": {
                "model_prompt": lambda user_id: self.get_model_prompt_ending(user_id),

                "choices": {"any": "opening_prompt"},
                "protocols": {"any": []}
                },

            "restart_prompt": {
                "model_prompt": lambda user_id: self.get_restart_prompt(user_id),

                "choices": {
                    "open_text": lambda user_id: self.determine_next_prompt_opening(user_id)
                },
                "protocols": {"open_text": []},
            },
        }
        self.QUESTION_KEYS = list(self.QUESTIONS.keys())
    
    ####################################################################
    ####            I N I T I A L I S E  C H A T B O T              ####
    ####  Clears chatbot parameters and prepares it for a new run   ####
    ####################################################################
    def create_new_run(self, user_id, db_session, user_session):
        new_run = UserModelRun(session_id=user_session.id)
        db_session.add(new_run)
        db_session.commit()
        self.current_run_ids[user_id] = new_run.id
        return new_run
    
    def clear_suggested_protocols(self):
        self.protocols_to_suggest = []

    def clear_language(self, user_id):
        self.language[user_id] = ""

    def clear_names(self, user_id):
        self.users_names[user_id] = ""

    def clear_datasets(self, user_id):
        self.datasets[user_id] = pd.DataFrame(columns=['sentences'])

    def clear_suggestions(self, user_id):
        self.suggestions[user_id] = []
        self.reordered_protocol_questions[user_id] = deque(maxlen=5)

    def clear_emotion_scores(self, user_id):
        self.guess_emotion_predictions[user_id] = ""

    def clear_choices(self, user_id):
        self.user_choices[user_id] = {"choices_made": {"current_choice": ""}}

    def initialise_remaining_choices(self, user_id):
        self.remaining_choices[user_id] = ["displaying_antisocial_behaviour", "internal_persecutor_saviour", "personal_crisis", "rigid_thought"]

    # def initialise_prev_questions(self, user_id):
    #     self.recent_questions[user_id] = []

    def set_language(self, user_id, language):
        '''
        Sets the language and extracts the dataset in the relevant language.
        - user_id [int]
        - language [str]: either "中文(ZH)" or "English(EN)"
        '''
        self.language[user_id] = language
        self.datasets[user_id] = pd.read_csv(f'{language}.csv', encoding='ISO-8859-1') 

    ##############################################################################
    ########             S E N T E N C E  P R O C E S S I N G             ########
    ########  Functions to extract and clean sentences before displaying  ########
    ##############################################################################
    def get_sentence(self, column_name, user_id):
        '''
        Extracts a sentence at random without replacement.
        - column_name [str]: the intended prompt (i.e. the column name)
        - user_id [int]: user's id

        Returns: 
        - sentence [str]: the selected prompt utterance.
        '''
        df = self.datasets[user_id]     # get the existing datafram for the user
        column = df[column_name]        # extract the relevant column

        # if the column is out of utterances, replenish list
        if not column.dropna().to_list():
            print(f'Sentences for {column_name} are now depleted. Replenishing sentences.')
            read_df = pd.read_csv(f'{self.language[user_id]}.csv', encoding='ISO-8859-1')
            df[column_name] = read_df[column_name]
            column = df[column_name]

        # select a random utterance from the list
        sentence = random.choice(column.dropna().to_list())

        # remove from the list to prevent calling the same ones
        df[column_name] = column[column!=sentence]

        return sentence

    def split_sentence(self, sentence):
        '''
        To make conversations easier to understand, we split each sentence into separate messages using this function.
        - sentence [string]: bot message to be shown to user.

        Returns:
        - the sentences to be outputed in separate segments [tuple]
        '''
        # split by punctuation
        temp_list = re.split('(?<=[.?!]) +', sentence)

        # remove any elements that are empty strings after the split
        if '' in temp_list:
            temp_list.remove('')

        return tuple(temp_list)

    #############################################################################
    ########                    G E T  P R O M P T S                     ########
    ########                 All functions return [str]                  ########
    #############################################################################
    def ask_feeling(self, user_id):
        '''
        Opening prompt: ask users how they feel.
        '''
        opening_prompt = {
            'English(EN)': "Hello, my name is Kai and I will be here to assist you today! How are you feeling today?",
            "中文(ZH)": "你好，我叫凯。我是您今天的助手！请问您今天感觉如何?"
        }
        
        return opening_prompt[self.language[user_id]]

    def get_restart_prompt(self, user_id):
        '''
        Restart prompt: ask users how they feel again.
        '''
        restart_prompt = {
            'English(EN)': "Please tell me again, how are you feeling today?",
            "中文(ZH)": "请您再告诉我您今天感觉如?"
        }
        
        return restart_prompt[self.language[user_id]]

    def get_model_prompt_project_emotion(self, user_id):
        '''
        Prompt to project emotions.
        '''
        prompt = {
            'English(EN)': "Thank you. While I have a think about which protocols would be best for you, please take your time now and try to project your current emotions onto your childhood self. When you are able to do this, please press 'continue' to receive your suggestions.",
            "中文(ZH)": "谢谢你，我现在会考虑哪种练习最适合你。请在这时候尝试把您现在的情绪投射到童年时期的自己上。当您做到这点时，请按按“继续。"
        }
        return self.split_sentence(prompt[self.language[user_id]])

    def get_model_prompt(self, user_id, column_name, special=False):
        '''
        Used by most self.QUESTIONS to get the prompt utterance.
        (specifically: base utterances 0-4, 6-12, 14, 16, 18-20)
        - user_id [int]
        - column_name [str]
        - special [bool]:   
            special prompts are those specific to negative emotions, requiring (self.user_emotions[user_id]) appended in front of column_name
            These include: ['personal_crisis', 'saviour', 'rigid_thoughts', 'internal_persecutor_victim', 'internal_persecutor_controlling',
                            'internal_persecutor_accusing', after_classification_negative', 'event_is_recent', 'revisiting_recent_events', 
                            'revisiting_distant_events', 'more_questions']
        '''
        if special:
            column_name = self.user_emotions[user_id] + column_name

        sentence = self.get_sentence(column_name, user_id)  # extract sentences from dataframe

        return self.split_sentence(sentence)                # split long sentences so it's more readable

    def get_model_prompt_antisocial(self, user_id):
        '''
        Get prompt for base utterance 6.
        '''
        other_emotions = {
            'English(EN)': "Envy, jealousy, greed, hatred, mistrust, malevolence, or revengefulness?",
            "中文(ZH)":"嫉妒、贪婪、仇恨、不信任、恶意或报复感"
        }
        column_name = self.user_emotions[user_id] + " - Have you strongly felt or expressed any of the following emotions towards someone:"
        question = self.get_sentence(column_name, user_id)
        
        return [self.split_sentence(question), other_emotions[self.language[user_id]]]

    def get_model_prompt_guess_emotion(self, user_id):
        '''
        Get prompt for base utterance 13.
        '''
        column_name = "All emotions - From what you have said I believe you are feeling {}. Is this correct?"
        my_string = self.get_sentence(column_name, user_id)
        question = my_string.format(self.guess_emotion_predictions[user_id].lower())    # places the emotions inside {}
        
        return self.split_sentence(question)

    def get_model_prompt_ending(self, user_id):
        '''
        Get prompt for base utterance 15.
        '''
        prompt = {
            "English(EN)": "You have been disconnected. Refresh the page if you would like to start over.",
            "中文(ZH)": "如果您改变了主意，想重新开始，请刷新网页。"
        }
        column_name = "All emotions - Thank you for taking part. See you soon"
        question = self.get_sentence(column_name, user_id)
        
        return [self.split_sentence(question), prompt[self.language[user_id]]]

    def get_model_prompt_trying_protocol(self, user_id):
        '''
        Get prompt for base utterance 17.
        '''
        prompt = {
            "English(EN)": "You have selected Protocol ",
            "中文(ZH)": "您选了协议"
        }
        column_name = "All emotions - Please try to go through this protocol now. When you finish, press 'continue'"
        question = self.get_sentence(column_name, user_id)
        
        return [prompt[self.language[user_id]] + str(self.current_protocol_ids[user_id][0]) + ". ", self.split_sentence(question)]

    
    ################################################################################
    ###             F O L L O W - U P  P R O M P T  H E A D E R S                ###
    ###     Triggered following choice selection. All functions return [str]     ###
    ################################################################################
    def get_after_classification(self, user_id, emotion):
        '''
        Triggered when emotions are predicted wrongly by the model and users are prompted to select the correct emotion they're feeling.
        Updates the user_emotion and guess_emotion_predictions 
        - user_id [int]
        - emotion [str]: must be one of the following ['Sad','Angry','Anxious/Scared','Happy/Content']

        Returns: "after_classification_negative" or "after_classification_positive" [str]
        '''
        self.guess_emotion_predictions[user_id] = emotion
        self.user_emotions[user_id] = emotion

        if emotion in ['Sad','Angry','Anxious/Scared']:
            return "after_classification_negative"
        elif emotion == 'Happy/Content':
            return "after_classification_positive"
        else:
            raise Exception(f"emotion {emotion} not in list of acceptable emotions ['Sad','Angry','Anxious/Scared','Happy/Content']")

    def determine_next_prompt_new_protocol(self, user_id):
        '''
        Triggered when patients choose "Yes (show follow-up suggestions)" when prompted to do a follow-up protocol.

        Returns: the next question prompt to proceed to [str]
        '''
        try:
            self.suggestions[user_id]
        except KeyError:
            self.suggestions[user_id] = []
        if len(self.suggestions[user_id]) > 0:
            return "suggestions"
        return "more_questions"

    def get_next_question(self, user_id):
        '''
        The additional question retrieval strategy, triggered when users give permission to ask additional questions.
        Selects question from a pool of questions (remaining_choices) at random.
        If pool exhausted, ask user to project_emotion.
        '''
        if self.remaining_choices[user_id] == []:
            return "project_emotion"

        selected_choice = np.random.choice(self.remaining_choices[user_id])
        self.remaining_choices[user_id].remove(selected_choice)
        return selected_choice

    def determine_next_prompt_opening(self, user_id):
        '''
        Triggered when user replies about how they feel.
        Updates guess_emotion_predictions and user_emotions dict.

        Returns: the next prompt [str] = "guess_emotion" 
        '''
        user_response = self.user_choices[user_id]["choices_made"]["ask_feeling"]
        emotion = get_emotion(user_response, self.language[user_id])
        self.guess_emotion_predictions[user_id] = emotion
        self.user_emotions[user_id] = emotion

        return "guess_emotion"

    ########################################################################
    ######                P R O C E S S  C H O I C E S                ######
    ###### Functions used to update variables following user decision ######
    ########################################################################
    def save_current_choice(self, user_id, user_choice, user_session, db_session):
        '''
        Triggered when frontend sends user's choice to the backend via update_session().
        (i) Saves the user choice in self.user_choices
        (ii) Saves the user choice in the database 
        - user_id [int]
        - user_choice [str]: the message written by user [str] 

        Returns: choice_made [Choice object]
        '''
        # Save current choice
        current_choice = self.user_choices[user_id]["choices_made"]["current_choice"]
        self.user_choices[user_id]["choices_made"][current_choice] = user_choice        # populate the value 
        # curr_prompt = self.QUESTIONS[current_choice]["model_prompt"]
        

        # User selected a protocol following protocol suggestions
        # user_choice [str]: either (i) title for the corr protocol, or (ii) protocol number itself
        if current_choice == "suggestions":
            # convert user_choice which is a string to an int
            try:
                current_protocol = self.TITLE_TO_PROTOCOL[user_choice]  # (i) if user_choice is the title for the corr protocol
            except KeyError:
                current_protocol = int(user_choice)                     # (ii) if user_choice is a str of the protocol number

            # Protocol object
            protocol_chosen = Protocol(
                protocol_chosen=current_protocol,
                user_id=user_id,
                session_id=user_session.id,
                run_id=self.current_run_ids[user_id],
            )

            # Add protocol
            db_session.add(protocol_chosen)
            db_session.commit()
            self.current_protocol_ids[user_id] = [current_protocol, protocol_chosen.id]

            for i in range(len(self.suggestions[user_id])):
                curr_protocols = self.suggestions[user_id][i]
                if curr_protocols[0] == self.PROTOCOL_TITLES[current_protocol]:
                    curr_protocols.popleft()
                    if len(curr_protocols) == 0:
                        self.suggestions[user_id].pop(i)
                    break

        # User selected ["Better", "Worse", "Neutral"] after asked if they found the protocol userful
        # user_choice [str] = "Better" or "Worse" or Neutral
        elif current_choice == "user_found_useful":
            current_protocol = Protocol.query.filter_by(
                id=self.current_protocol_ids[user_id][1]
            ).first()
            current_protocol.protocol_was_useful = user_choice
            db_session.commit()

        # User selected if the model predicted their emotions correctly.
        # user_choice [str] = 'yes','no'
        if current_choice == "guess_emotion":
            option_chosen = user_choice + " ({})".format(
                self.guess_emotion_predictions[user_id]
            )
        else:
            option_chosen = user_choice

        choice_made = Choice(
            choice_desc=current_choice,
            option_chosen=option_chosen,
            user_id=user_id,
            session_id=user_session.id,
            run_id=self.current_run_ids[user_id],
        )

        db_session.add(choice_made)
        db_session.commit()

        return choice_made
    
    def determine_next_choice(self, user_id, input_type, user_choice):
        '''
        Actions the follow-up prompt.

        Returns [dict]: {"model_prompt": next_prompt, "choices": next_choices}
        '''

        current_choice = self.user_choices[user_id]["choices_made"]["current_choice"]   # the current prompt
        current_choice_for_question = self.QUESTIONS[current_choice]["choices"]         # the choices available for the current prompt
        current_protocols = self.QUESTIONS[current_choice]["protocols"]                 # the protocols available for the current prompt

        # If it's a button selection
        if input_type != "open_text":
            if (
                current_choice != "suggestions"
                and current_choice != "event_is_recent"
                and current_choice != "more_questions"
                and current_choice != "after_classification_positive"
                and current_choice != "user_found_useful"
                and current_choice != "check_emotion"
                and current_choice != "new_protocol_better"
                and current_choice != "new_protocol_worse"
                and current_choice != "project_emotion"
                and current_choice != "after_classification_negative"
            ):
                user_choice = user_choice.lower()

            if (
                current_choice == "suggestions"
            ):
                try:
                    current_protocol = self.TITLE_TO_PROTOCOL[user_choice]
                except KeyError:
                    current_protocol = int(user_choice)
                protocol_choice = self.PROTOCOL_TITLES[current_protocol]
                next_choice = current_choice_for_question[protocol_choice]
                protocols_chosen = current_protocols[protocol_choice]

            # Users to choose the emotions they are actually feeling after wrong predictions
            # user_choice = Sad, Angry, Anxious/Scared, Happy/Content
            elif current_choice == "check_emotion":
                if user_choice in ["Sad","Angry","Anxious/Scared","Happy/Content"]:
                    next_choice = current_choice_for_question[user_choice]
                    protocols_chosen = current_protocols[user_choice]
                
                else:
                    raise Exception(f'user_choice should only contain ["Sad","Angry","Anxious/Scared","Happy/Content"] but received {user_choice} instead.')
            
            else:
                next_choice = current_choice_for_question[user_choice]
                protocols_chosen = current_protocols[user_choice]

        # for open_text answers (i.e when ask_feeling)
        else:
            next_choice = current_choice_for_question["open_text"]  # for ask_feeling, it's the determine_next_opening_prompt function
            protocols_chosen = current_protocols["open_text"]       # for ask_feeling, it's []

        # get the next_choice in string form, if it's a function
        if callable(next_choice):
            next_choice = next_choice(user_id) 

        if current_choice == "guess_emotion" and user_choice.lower() == "yes":
            # for guess_emotion, next_choice is a dict
            if self.guess_emotion_predictions[user_id] in ["Sad","Angry","Anxious/Scared","Happy/Content"]:
                next_choice = next_choice[self.guess_emotion_predictions[user_id]]
            
            else:
                raise Exception(f'self.guess_emotion_predictions[user_id] should only contain ["Sad","Angry","Anxious/Scared","Happy/Content"] but received {self.guess_emotion_predictions[user_id]} instead.')

        # not used i think 
        if callable(protocols_chosen):
            protocols_chosen = protocols_chosen(user_id)

        # get the next model prompt ([str] or <function>)
        next_prompt = self.QUESTIONS[next_choice]["model_prompt"]               
        if callable(next_prompt):
            next_prompt = next_prompt(user_id)   # if function, get the string prompt
        
        if (len(protocols_chosen) > 0 and current_choice != "suggestions"):
            self.update_suggestions(user_id, protocols_chosen)

        # Case: new suggestions being created after first protocol attempted
        # if next_choice == "ask_feeling":
        #     self.clear_suggestions(user_id)
        #     self.clear_emotion_scores(user_id)
        #     self.create_new_run(user_id, db_session, user_session)

        if next_choice == "suggestions":
            next_choices = self.get_suggestions(user_id)

        else:
            next_choices = list(self.QUESTIONS[next_choice]["choices"].keys())          # [yes, no]

        self.user_choices[user_id]["choices_made"]["current_choice"] = next_choice

        return {"model_prompt": next_prompt, "choices": next_choices}

    ########################################################################
    ######           S U G G E S T I O N  F U N C T I O N S           ######
    ######    Functions relating to compiling protocol suggestions    ######
    ########################################################################
    def get_suggestions(self, user_id):
        '''
        Function puts together a lists of protocols to suggest to the user.
        Based on protocols collected at each step of the dialogue + adds some if less that 4 protocols suggested.
        - user_id [int]
        
        Returns: suggestions [list]
        '''
        suggestions = []
        for curr_suggestions in list(self.suggestions[user_id]):
            if len(curr_suggestions) > 2:
                i, j = random.choices(range(len(curr_suggestions)), k=2)
                # weeds out some gibberish 
                if curr_suggestions[i] and curr_suggestions[j] in self.PROTOCOL_TITLES: 
                    suggestions.extend([curr_suggestions[i], curr_suggestions[j]])
            else:
                suggestions.extend(curr_suggestions)
            suggestions = set(suggestions)
            suggestions = list(suggestions)

        # if there's less than 4 suggestions, add random protocols (except protocol 6 & 11) w/o repetition 
        while len(suggestions) < 4: 
            p = random.choice([i for i in range(1,20) if i not in [6,11]]) 
            if (any(self.PROTOCOL_TITLES[p] not in curr_suggestions for curr_suggestions in list(self.suggestions[user_id]))
                and self.PROTOCOL_TITLES[p] not in self.recent_protocols and self.PROTOCOL_TITLES[p] not in suggestions):
                        suggestions.append(self.PROTOCOL_TITLES[p])
                        self.suggestions[user_id].extend([self.PROTOCOL_TITLES[p]])
        return suggestions
    
    def update_suggestions(self, user_id, protocols):
        ''''
        Update the suggestion pool.
        '''
        # Check if user_id already has suggestions
        try:
            self.suggestions[user_id]
        except KeyError:
            self.suggestions[user_id] = []

        if type(protocols) != list:
            self.suggestions[user_id].append(deque([protocols]))
        else:
            self.suggestions[user_id].append(deque(protocols))



 
# not sure if this is needed
    # def update_conversation(self, user_id, new_dialogue, db_session, app):
    #     try:
    #         session_id = self.user_choices[user_id]["current_session_id"]
    #         curr_session = UserModelSession.query.filter_by(id=session_id).first()
    #         if curr_session.conversation is None:
    #             curr_session.conversation = "" + new_dialogue
    #         else:
    #             curr_session.conversation = curr_session.conversation + new_dialogue
    #         curr_session.last_updated = datetime.datetime.utcnow()
    #         db_session.commit()
    #     except KeyError:
    #         curr_session = UserModelSession(
    #             user_id=user_id,
    #             conversation=new_dialogue,
    #             last_updated=datetime.datetime.utcnow(),
    #         )

    #         db_session.add(curr_session)
    #         db_session.commit()
    #         self.user_choices[user_id]["current_session_id"] = curr_session.id

# taken out of save_decision
        # if callable(curr_prompt):
        #     curr_prompt = curr_prompt(user_id, db_session, user_session, app)

        # else:
        #     self.update_conversation(
        #         user_id,
        #         "Model:{} \nUser:{} \n".format(curr_prompt, user_choice),
        #         db_session,
        #         app,
        #     )