from lib2to3.pgen2.pgen import DFAState
from locale import D_FMT
from tkinter import E
import nltk
from regex import P
from model.models import UserModelSession, Choice, UserModelRun, Protocol
import pandas as pd
import numpy as np
import random
from collections import deque
import re
import datetime
import time
from titles import titles
from utils import get_sentence, split_sentence, get_emotion


class ModelDecisionMaker:
    def __init__(self):

        self.recent_protocols = deque(maxlen=20)                # recent protocols
        self.reordered_protocol_questions = {}                  # reordered protocol questions
        # self.protocols_to_suggest = []
        self.current_protocol_ids = {}                          # current protocol selected by user
        # self.current_protocols = {}                             
        self.positive_protocols = [i for i in range(1, 21)]     # list of all possible protocols
        self.remaining_choices = {}                             # tracks remaining protocol choices
        self.PROTOCOL_TITLES = {}                               # List of protocol titles in the language tbe user selected (related >> see titles.py)
        self.TITLE_TO_PROTOCOL = {}                             # Dict of protocol titles to its numbering 
        self.current_run_ids = {}                               # Protocol run id
        self.user_choices = {}                                  # Selections made by users (buttons clicked)
        self.suggestions = {}                                   # Protocol suggestions for users 
        self.user_emotions = {}                                 # User's actual emotions
        self.guess_emotion_predictions = {}                     # Emotion prediction made by model
        self.language = {}                                      # User's language choice selection
        self.datasets = {}                                      # Dataset containing utterances
        
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
                    "English(EN)": {
                        "yes": {
                            "Sad": "after_classification_negative",
                            "Angry": "after_classification_negative",
                            "Anxious/Scared": "after_classification_negative",
                            "Happy/Content": "after_classification_positive",
                            },
                        "no": "check_emotion",
                    },
                    "中文(ZH)": {
                        "对，这是正确的": {
                            "悲伤": "after_classification_negative",
                            "愤怒": "after_classification_negative",
                            "焦虑": "after_classification_negative",
                            "快乐": "after_classification_positive",
                        },
                        "不正确": "check_emotion"
                    }
                },
                "protocols": {
                    "English(EN)": {
                        "yes": [],
                        "no": []
                    },
                    "中文(ZH)": {
                        "对，这是正确的": [],
                        "不正确": []
                    }
                }
            },

            "check_emotion": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name = "All emotions - I am sorry. Please select from the emotions below the one that best reflects what you are feeling:"
                    ),

                "choices": {
                    "English(EN)": {
                        "Sad": lambda user_id: self.get_after_classification(user_id, emotion='Sad'),
                        "Angry": lambda user_id: self.get_after_classification(user_id, emotion='Angry'),
                        "Anxious/Scared": lambda user_id: self.get_after_classification(user_id, emotion='Anxious/Scared'),
                        "Happy/Content": lambda user_id: self.get_after_classification(user_id, emotion="Happy/Content"),
                    },
                    "中文(ZH)": {
                        "悲伤": lambda user_id: self.get_after_classification(user_id, emotion='Sad'),
                        "愤怒": lambda user_id: self.get_after_classification(user_id, emotion='Angry'),
                        "焦虑": lambda user_id: self.get_after_classification(user_id, emotion='Anxious/Scared'),
                        "快乐": lambda user_id: self.get_after_classification(user_id, emotion="Happy/Content")
                    }
                },
                "protocols": {
                    "English(EN)": {
                        "Sad": [],
                        "Angry": [],
                        "Anxious/Scared": [],
                        "Happy/Content": []
                    },
                    "中文(ZH)":{
                        "悲伤": [],
                        "愤怒": [], 
                        "焦虑": [],
                        "快乐": []
                    }
                }
            },
            
            ############# NEGATIVE EMOTIONS (SADNESS, ANGER, FEAR/ANXIETY) #############
            "after_classification_negative": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name=" - Was this caused by a specific event/s?",
                    special=True
                    ),
                "choices": {
                    "English(EN)": {
                        "Yes, something happened": "event_is_recent",
                        "No": "more_questions",
                    },
                    "中文(ZH)":{
                        "是的": "event_is_recent",
                        "不是": "more_questions"
                    }
                },
                "protocols": {
                    "English(EN)": {
                        "Yes, something happened": [],
                        "No": []
                    },
                    "中文(ZH)":{
                        "是的": [],
                        "不是": []
                    },
                },
            },

            "event_is_recent": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name=" - Was this caused by a recent or distant event (or events)?",
                    special=True
                    ),

                "choices": {
                    "English(EN)": {
                        "It was recent": "revisiting_recent_events",
                        "It was distant": "revisiting_distant_events",
                    }, 
                    "中文(ZH)":{
                        "最近发生的": "revisiting_recent_events",
                        "以前发生的": "revisiting_distant_events",
                    }
                },
                "protocols": {
                    "English(EN)": {
                        "It was recent": [],
                        "It was distant": []
                    }, 
                    "中文(ZH)":{
                        "最近发生的": [],
                        "以前发生的": []
                    }
                },
            },

            "revisiting_recent_events": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name=" - Have you recently attempted protocol 11 and found this reignited unmanageable emotions as a result of old events?",
                    special=True
                    ),
                "choices": {
                    "English(EN)": {
                        "yes": "more_questions",
                        "no": "more_questions",
                    }, 
                    "中文(ZH)":{
                        "有": "more_questions",
                        "没有": "more_questions",
                    }
                },
                "protocols": {
                    "English(EN)": {
                        "yes": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[7,8]
                        ),
                        "no": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[11]
                        ),
                    }, 
                    "中文(ZH)":{
                        "有": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[7,8]
                        ),
                        "没有": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[11]
                        ),
                    }
                },
            },

            "revisiting_distant_events": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name=" - Have you recently attempted protocol 6 and found this reignited unmanageable emotions as a result of old events?",
                    special=True
                    ),
                "choices": {
                    "English(EN)": {
                        "yes": "more_questions",
                        "no": "more_questions",
                    }, 
                    "中文(ZH)":{
                        "有": "more_questions",
                        "没有": "more_questions",
                    }
                },
                "protocols": {
                    "English(EN)": {
                        "yes": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[13, 17]
                        ), 
                        "no": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[6]
                        ), 
                    }, 
                    "中文(ZH)":{
                        "有":lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[13, 17]
                        ), 
                        "没有": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[6]
                        ), 
                    }
                },
            },

            "more_questions": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id,
                    column_name=" - Thank you. Now I will ask some questions to understand your situation.",
                    special=True
                    ),

                "choices": {
                    "English(EN)": {
                        "Okay": lambda user_id: self.get_next_question(user_id),
                        "I'd rather not": "project_emotion",
                    }, 
                    "中文(ZH)":{
                        "请继续（详细问题）": lambda user_id: self.get_next_question(user_id),
                        "我不想再继续": "project_emotion",
                    }
                },
                "protocols": {
                    "English(EN)": {
                        "Okay": [],
                        "I'd rather not": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[13]
                        ), 
                    }, 
                    "中文(ZH)":{
                        "请继续（详细问题）": [],
                        "我不想再继续": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[13]
                        ), 
                    }
                },
            },

            "displaying_antisocial_behaviour": {
                "model_prompt": lambda user_id : self.get_model_prompt_antisocial(user_id),

                "choices": {
                    "English(EN)": {
                        "yes": "project_emotion",
                        "no": lambda user_id: self.get_next_question(user_id),
                    }, 
                    "中文(ZH)":{
                        "是": "project_emotion",
                        "否": lambda user_id: self.get_next_question(user_id),
                    }
                },
                "protocols": {
                    "English(EN)": {
                        "yes": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[13, 14]
                        ), 
                        "no": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[13]
                        ), 
                    }, 
                    "中文(ZH)":{
                        "是":lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[13, 14]
                        ), 
                        "否":lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[13]
                        ), 
                    }
                },
            },

            "internal_persecutor_saviour": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name=" - Do you believe that you should be the saviour of someone else?",
                    special=True
                    ),
                "choices": {
                    "English(EN)": {
                        "yes": "project_emotion",
                        "no": "internal_persecutor_victim",
                    }, 
                    "中文(ZH)":{
                        "是":"project_emotion",
                        "否":"internal_persecutor_victim",
                    }
                },
                "protocols": {
                    "English(EN)": {
                        "yes": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[8, 15, 16, 19]
                        ), 
                        "no": []
                    }, 
                    "中文(ZH)":{
                        "是": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[8, 15, 16, 19]
                        ), 
                        "否": []
                    }
                },
            },

            "internal_persecutor_victim": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name=" - Do you see yourself as the victim, blaming someone else for how negative you feel?",
                    special=True
                    ),
                "choices": {
                    "English(EN)": {
                        "yes": "project_emotion",
                        "no": "internal_persecutor_controlling",
                    }, 
                    "中文(ZH)":{
                        "是": "project_emotion",
                        "否": "internal_persecutor_controlling",
                    }
                },
                "protocols": {
                    "English(EN)": {
                        "yes": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[8, 15, 16, 19]
                        ), 
                        "no": []
                    },
                    "中文(ZH)":{
                        "是": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[8, 15, 16, 19]
                        ), 
                        "否": []
                    }
                },
            },

            "internal_persecutor_controlling": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name=" - Do you feel that you are trying to control someone?",
                    special=True
                    ),

                "choices": {
                    "English(EN)": {
                        "yes": "project_emotion",
                        "no": "internal_persecutor_accusing"
                    }, 
                    "中文(ZH)":{
                        "是": "project_emotion",
                        "否": "internal_persecutor_accusing"
                    }
                },
                "protocols": {
                     "English(EN)": {
                        "yes": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[8, 15, 16, 19]
                        ), 
                        "no": []
                    }, 
                    "中文(ZH)":{
                        "是": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[8, 15, 16, 19]
                        ), 
                        "否": []
                    }
                },
            },

            "internal_persecutor_accusing": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name=" - Are you always blaming and accusing yourself for when something goes wrong?",
                    special=True
                    ),
                "choices": {
                     "English(EN)": {
                        "yes": "project_emotion",
                        "no": lambda user_id: self.get_next_question(user_id),
                    }, 
                    "中文(ZH)":{
                        "是": "project_emotion",
                        "否":lambda user_id: self.get_next_question(user_id),
                    }
                },
                "protocols": {
                    "English(EN)": {
                        "yes": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[8, 15, 16, 19]
                        ), 
                        "no": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[13]
                        ),
                    }, 
                    "中文(ZH)":{
                        "是": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[8, 15, 16, 19]
                        ), 
                        "否": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[13]
                        ), 
                    }
                },
            },

            "rigid_thought": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id,
                    column_name=" - In previous conversations, have you considered other viewpoints presented?",
                    special=True
                    ),

                "choices": {
                    "English(EN)": {
                        "yes": lambda user_id: self.get_next_question(user_id),
                        "no": "project_emotion",

                    }, 
                    "中文(ZH)":{
                        "是": lambda user_id: self.get_next_question(user_id),
                        "否": "project_emotion",
                    }
                },
                "protocols": {
                    "English(EN)": {
                        "yes": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[13]
                        ), 
                        "no": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[13, 19]
                        ), 
                    }, 
                    "中文(ZH)":{
                        "是": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[13]
                        ), 
                        "否": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[13, 19]
                        ), 
                    }
                },
            },


            "personal_crisis": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name=" - Are you undergoing a personal crisis (experiencing difficulties with loved ones e.g. falling out with friends)?",
                    special=True
                    ),

                "choices": {
                    "English(EN)": {
                        "yes": "project_emotion",
                        "no": lambda user_id: self.get_next_question(user_id),
                    }, 
                    "中文(ZH)":{
                        "是": "project_emotion",
                        "否": lambda user_id: self.get_next_question(user_id),
                    }
                },
                "protocols": {
                    "English(EN)": {
                        "yes": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[13, 17]
                        ), 
                        "no": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[13]
                        ), 
                    }, 
                    "中文(ZH)":{
                        "是": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[13, 17]
                        ), 
                        "否": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[13]
                        ), 
                    }
                },
            },

            ################# POSITIVE EMOTION (HAPPINESS/CONTENT) #################

            "after_classification_positive": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name="Happy - That's Good! Let me recommend a protocol you can attempt."
                    ),

                "choices": {
                    "English(EN)": {
                        "Okay": "suggestions",
                        "No, thank you": "ending_prompt"
                    }, 
                    "中文(ZH)":{
                        "好的，我想尝试一些协议": "suggestions",
                        "不用了": "ending_prompt"
                    }
                },
                "protocols": {
                    "English(EN)": {
                        "Okay": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[9, 10, 11]
                        ), 
                        "No, thank you": []
                    }, 
                    "中文(ZH)":{
                        "好的，我想尝试一些协议": lambda user_id: self.get_protocols(
                            user_id, 
                            protocol_num=[9, 10, 11]
                        ), 
                        "不用了": []
                    }
                },
            },

            ############################# ALL EMOTIONS #############################

            "project_emotion": {
               "model_prompt": lambda user_id: self.get_model_prompt_project_emotion(user_id),

               "choices": {
                    "English(EN)": {
                        "Continue": "suggestions",
                    }, 
                    "中文(ZH)":{
                        "继续": "suggestions",
                    }
               },
               "protocols": {
                    "English(EN)": {
                        "Continue": [],
                    }, 
                    "中文(ZH)":{
                        "继续": [],
                    }
               },
            },

            "suggestions": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name= "All emotions - Here are my recommendations, please select the protocol that you would like to attempt"
                    ),

                "choices": lambda user_id: self.get_suggestion_choices(user_id),

                "protocols": lambda user_id: self.get_suggestion_protocols(user_id),
            },

            "trying_protocol": {
                "model_prompt": lambda user_id: self.get_model_prompt_trying_protocol(user_id),

                "choices": {
                    "English(EN)": {
                        "continue": "user_found_useful"
                    }, 
                    "中文(ZH)":{
                        "继续": "user_found_useful"
                    }
                },
                "protocols": {
                    "English(EN)": {
                        "continue": []
                    }, 
                    "中文(ZH)":{
                        "继续": []
                    }
                },
            },

            "user_found_useful": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name = "All emotions - Do you feel better or worse after having taken this protocol?"
                    ),
                "choices": {
                    "English(EN)": {
                        "I feel better": "new_protocol_better",
                        "I feel worse": "new_protocol_worse",
                        "Neutral": lambda user_id: self.get_neutral_prompt(user_id)
                    }, 
                    "中文(ZH)":{
                        "我觉得好点":"new_protocol_better",
                        "我觉得更糟":"new_protocol_worse",
                        "不变": lambda user_id: self.get_neutral_prompt(user_id)
                    }
                },
                "protocols": {
                    "English(EN)": {
                        "I feel better": [],
                        "I feel worse": [],
                        "Neutral": [],
                    }, 
                    "中文(ZH)":{
                        "我觉得好点": [],
                        "我觉得更糟": [],
                        "不变": []
                    }
                },
            },

            "new_protocol_better": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name="All emotions - Would you like to attempt another protocol? (Patient feels better)"
                    ),
                "choices": {
                    "English(EN)": {
                        "Yes (show follow-up suggestions)": lambda user_id: self.determine_next_prompt_new_protocol(user_id),
                        "Yes (restart questions)": "restart_prompt",
                        "No (end session)": "ending_prompt",
                    }, 
                    "中文(ZH)":{
                        "是（其它协议）":lambda user_id: self.determine_next_prompt_new_protocol(user_id),
                        "是（从头开始）":"restart_prompt",
                        "否（结束对话）":"ending_prompt",
                    }
                },
                "protocols": {
                    "English(EN)": {
                        "Yes (show follow-up suggestions)": [],
                        "Yes (restart questions)": [],
                        "No (end session)": []
                    }, 
                    "中文(ZH)":{
                        "是（其它协议）": [],
                        "是（从头开始）": [],
                        "否（结束对话）": []
                    }
                },
            },

            "new_protocol_worse": {
                "model_prompt": lambda user_id: self.get_model_prompt(
                    user_id, 
                    column_name="All emotions - Would you like to attempt another protocol? (Patient feels worse)"
                    ),
                "choices": {
                    "English(EN)": {
                        "Yes (show follow-up suggestions)": lambda user_id: self.determine_next_prompt_new_protocol(user_id),
                        "Yes (restart questions)": "restart_prompt",
                        "No (end session)": "ending_prompt",
                    }, 
                    "中文(ZH)":{
                        "是（其它协议）": lambda user_id: self.determine_next_prompt_new_protocol(user_id),
                        "是（从头开始）": "restart_prompt",
                        "否（结束对话）": "ending_prompt",
                    }
                },
                "protocols": {
                    "English(EN)": {
                        "Yes (show follow-up suggestions)": [],
                        "Yes (restart questions)": [],
                        "No (end session)": []
                    }, 
                    "中文(ZH)":{
                        "是（其它协议）": [],
                        "是（从头开始）": [],
                        "否（结束对话）": []
                    }
                },
            },

            "ending_prompt": {
                "model_prompt": lambda user_id: self.get_model_prompt_ending(user_id),

                "choices": {
                    "any": "opening_prompt"
                },
                "protocols": {
                    "any": []
                }
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
    
    # def clear_suggested_protocols(self):
    #     self.protocols_to_suggest = []

    def clear_language(self, user_id):
        self.language[user_id] = ""

    def clear_datasets(self, user_id):
        self.datasets[user_id] = pd.DataFrame(columns=['sentences'])

    def clear_suggestions(self, user_id):
        self.suggestions[user_id] = []
        self.reordered_protocol_questions[user_id] = deque(maxlen=5)

    def clear_emotion_predictions(self, user_id):
        self.guess_emotion_predictions[user_id] = ""

    def clear_choices(self, user_id):
        self.user_choices[user_id] = {"choices_made": {"current_choice": ""}}

    def initialise_remaining_choices(self, user_id):
        self.remaining_choices[user_id] = ["displaying_antisocial_behaviour", "internal_persecutor_saviour", "personal_crisis", "rigid_thought"]

    def set_language(self, user_id, language):
        '''
        Sets the language and extracts the dataset + protocol titles in the relevant language.
        - user_id [int]
        - language [str]: either "中文(ZH)" or "English(EN)"
        '''
        self.language[user_id] = language
        self.datasets[user_id] = pd.read_csv(f'utterances/{language}.csv') 

        self.PROTOCOL_TITLES[user_id] = titles[language]

        self.TITLE_TO_PROTOCOL[user_id] = {self.PROTOCOL_TITLES[user_id][i]: i for i in range(len(self.PROTOCOL_TITLES[user_id]))}
    
    def get_suggestion_choices(self, user_id):
        return {self.PROTOCOL_TITLES[user_id][k]: "trying_protocol" for k in self.positive_protocols}
    
    def get_suggestion_protocols(self,user_id):
        return {self.PROTOCOL_TITLES[user_id][k]: [self.PROTOCOL_TITLES[user_id][k]] for k in self.positive_protocols}

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
        return split_sentence(prompt[self.language[user_id]])

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
            if self.user_emotions[user_id] in ['悲伤','愤怒','焦虑']:
                zh2en = {'悲伤':'Sad', '愤怒':'Angry', '焦虑':"Anxious/Scared"}
                column_name = zh2en[self.user_emotions[user_id]] + column_name
            else:
                column_name = self.user_emotions[user_id] + column_name

        sentence = get_sentence(column_name=column_name, dataset=self.datasets[user_id], language=self.language[user_id])  # extract sentences from dataframe

        return split_sentence(sentence)                # split long sentences so it's more readable

    def get_model_prompt_antisocial(self, user_id):
        '''
        Get prompt for base utterance 6.
        '''
        other_emotions = {
            'English(EN)': "Envy, jealousy, greed, hatred, mistrust, malevolence, or revengefulness?",
            "中文(ZH)":"嫉妒、贪婪、仇恨、不信任、恶意或报复感"
        }
        column_name = self.user_emotions[user_id] + " - Have you strongly felt or expressed any of the following emotions towards someone:"
        question = get_sentence(column_name=column_name, dataset=self.datasets[user_id], language=self.language[user_id])
        
        return [split_sentence(question), other_emotions[self.language[user_id]]]

    def get_model_prompt_guess_emotion(self, user_id):
        '''
        Get prompt for base utterance 13.
        '''
        column_name = "All emotions - From what you have said I believe you are feeling {}. Is this correct?"
        my_string = get_sentence(column_name=column_name, dataset=self.datasets[user_id], language=self.language[user_id])
        question = my_string.format(self.guess_emotion_predictions[user_id].lower())    # places the emotions inside {}
        
        return split_sentence(question)

    def get_model_prompt_ending(self, user_id):
        '''
        Get prompt for base utterance 15.
        '''
        prompt = {
            "English(EN)": "You have been disconnected. Refresh the page if you would like to start over.",
            "中文(ZH)": "如果您改变了主意，想重新开始，请刷新网页。"
        }
        column_name = "All emotions - Thank you for taking part. See you soon"
        question = get_sentence(column_name=column_name, dataset=self.datasets[user_id], language=self.language[user_id])
        
        return [split_sentence(question), prompt[self.language[user_id]]]

    def get_model_prompt_trying_protocol(self, user_id):
        '''
        Get prompt for base utterance 17.
        '''
        prompt = {
            "English(EN)": "You have selected Protocol ",
            "中文(ZH)": "您选了协议"
        }
        column_name = "All emotions - Please try to go through this protocol now. When you finish, press 'continue'"
        question = get_sentence(column_name=column_name, dataset=self.datasets[user_id], language=self.language[user_id])
        
        return [prompt[self.language[user_id]] + str(self.current_protocol_ids[user_id][0]) + ". ", split_sentence(question)]

    
    ################################################################################
    ###             F O L L O W - U P  P R O M P T  H E A D E R S                ###
    ###     Triggered following choice selection. All functions return [str]     ###
    ################################################################################
    def get_after_classification(self, user_id, emotion):
        '''
        Triggered when emotions are predicted wrongly by the model and users are prompted to select the correct emotion they're feeling.
        (i) Updates the user_emotion and guess_emotion_predictions 
        (ii) Returns the next prompt heading
        - user_id [int]
        - emotion [str]: must be one of the following ['Sad','Angry','Anxious/Scared','Happy/Content'] in EN only 

        Returns [str]: "after_classification_negative" or "after_classification_positive" 
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

        Returns: "suggestions" or "more_questions"
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

        Returns: "project_emotion" or a choice from remaining_choices list
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

        Returns: "guess_emotion"
        '''
        user_response = self.user_choices[user_id]["choices_made"]["ask_feeling"]
        emotion = get_emotion(user_response, self.language[user_id])
                  
        self.guess_emotion_predictions[user_id] = emotion
        self.user_emotions[user_id] = emotion

        return "guess_emotion"
    
    def get_neutral_prompt(self, user_id):
        '''
        Triggered when users select that they feel "neutral" after attemping a protocol.

        Returns: "new_protocol_better" or "new_protocol_worse"
        '''
        if self.user_emotions[user_id] in ['Happy/Content','快乐']:
            return "new_protocol_better"
        elif self.user_emotions[user_id] in ['Sad', 'Anxious/Scared','Angry','焦虑','悲伤','愤怒']:
            return "new_protocol_worse"
        else:
            raise Exception(f"{self.user_emotions[user_id]} not in acceptable lists of emotions ['Happy/Content','快乐','Sad', 'Anxious/Scared','Angry','焦虑','悲伤','愤怒']")

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

        # User selected a protocol following protocol suggestions
        # user_choice [str]: either (i) title for the corr protocol, or (ii) protocol number itself
        if current_choice == "suggestions":
            # convert user_choice which is a string to an int
            try:
                current_protocol = self.TITLE_TO_PROTOCOL[user_id][user_choice]  # (i) if user_choice is the title for the corr protocol
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
                if curr_protocols[0] == self.PROTOCOL_TITLES[user_id][current_protocol]:
                    curr_protocols.popleft()
                    if len(curr_protocols) == 0:
                        self.suggestions[user_id].pop(i)
                    break

        # User selected ["Better", "Worse", "Neutral"] after asked if they found the protocol userful
        # user_choice [str] = "I feel better" or "I feel worse" or "Neutral"
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
        print("curr_choice", current_choice)
        # current_choice_for_question: the choices available for the current prompt
        try:
            current_choice_for_question = self.QUESTIONS[current_choice]["choices"][self.language[user_id]]  # if buttons, will have language key    
        except KeyError:
            print(f'{current_choice} does not have a language key in choices')
            current_choice_for_question = self.QUESTIONS[current_choice]["choices"]                          # if open_text or protocol buttons, no lanugage key
        except TypeError:
            print(f'{current_choice} is a function')
            current_choice_for_question = self.QUESTIONS[current_choice]["choices"](user_id)

        # current_protocols: the protocols available for the current prompt
        try:
            current_protocols = self.QUESTIONS[current_choice]["protocols"][self.language[user_id]]     
        except KeyError:
            print(f'{current_choice} does not have a language key in protocols')
            current_protocols = self.QUESTIONS[current_choice]["protocols"]
        except TypeError:
            print(f'{current_choice} is a function')
            current_protocols = self.QUESTIONS[current_choice]["protocols"](user_id)

        # Get the next_choice (i.e next prompt_heading) and protocols_chosen
        # If it's a button selection
        if input_type != "open_text":
            if current_choice not in ["suggestions", "event_is_recent", "more_questions", "after_classification_positive", "user_found_useful", "check_emotion", "new_protocol_better", "new_protocol_worse", "project_emotion", "after_classification_negative"]:
                user_choice = user_choice.lower()

            if current_choice == "suggestions":
                try:
                    current_protocol = self.TITLE_TO_PROTOCOL[user_id][user_choice]
                except KeyError:
                    current_protocol = int(user_choice)

                protocol_choice = self.PROTOCOL_TITLES[user_id][current_protocol]
                next_choice = current_choice_for_question[protocol_choice]
                protocols_chosen = current_protocols[protocol_choice]

            # elif current_choice == "check_emotion":  
            #     if user_choice not in ["Sad", "Angry", "Anxious/Scared", "Happy/Content", "悲伤", "愤怒", "快乐", "焦虑"]:
            #         raise Exception(f'user_choice should only contain ["Sad","Angry","Anxious/Scared","Happy/Content", "悲伤", "愤怒", "快乐", "焦虑"] but received {user_choice} instead.')

            #     next_choice = current_choice_for_question[user_choice]
            #     protocols_chosen = current_protocols[user_choice]
            
            else:
                next_choice = current_choice_for_question[user_choice]
                protocols_chosen = current_protocols[user_choice]

        # if open text
        else:
            next_choice = current_choice_for_question["open_text"]  # determine_next_prompt_opening() 
            protocols_chosen = current_protocols["open_text"]       # []
        
        # for guess_emotion, next_choice is a dict and needs to be referenced
        if current_choice == "guess_emotion" and user_choice.lower() in ["yes","对，这是正确的"]:
            if self.guess_emotion_predictions[user_id] in ["Sad","Angry","Anxious/Scared","Happy/Content", "悲伤", "愤怒", "快乐", "焦虑"]:
                next_choice = next_choice[self.guess_emotion_predictions[user_id]]

            else:
                raise Exception(f'self.guess_emotion_predictions[user_id] should only contain ["Sad","Angry","Anxious/Scared","Happy/Content", "悲伤", "愤怒", "快乐", "焦虑"] but contains {self.guess_emotion_predictions[user_id]}.')

        # get the next_choice in string form, if it's a function
        if callable(next_choice):
            next_choice = next_choice(user_id) 

        print("next_choice", next_choice)

        # get the protocols_chosen in string form, if it's a function 
        if callable(protocols_chosen):
            protocols_chosen = protocols_chosen(user_id)

        # Get the next prompt ([str] or <function>)
        next_prompt = self.QUESTIONS[next_choice]["model_prompt"]           
        if callable(next_prompt):
            next_prompt = next_prompt(user_id)   # if function, get the string prompt

        # Update protocol suggestions
        if (len(protocols_chosen) > 0 and current_choice != "suggestions"):
            self.update_suggestions(user_id, protocols_chosen)

        # Get the next prompt's choices
        if next_choice == "suggestions":
            next_choices = self.get_suggestions(user_id)

        else:
            try:
                next_choices = list(self.QUESTIONS[next_choice]["choices"][self.language[user_id]].keys())          
            # when next_choice's "choices" does not contain a language key
            except KeyError:
                next_choices = list(self.QUESTIONS[next_choice]["choices"].keys())  

        self.user_choices[user_id]["choices_made"]["current_choice"] = next_choice

        return {"model_prompt": next_prompt, "choices": next_choices}

    ########################################################################
    ######           S U G G E S T I O N  F U N C T I O N S           ######
    ######    Functions relating to compiling protocol suggestions    ######
    ########################################################################
    def get_protocols(self, user_id, protocol_num):
        '''
        Function initialises the protocols for a given prompt.
        - user_id [int]
        - protocols [list of ints]: lists the protocols required

        Returns: list of protocol names
        '''
        protocol_titles = self.PROTOCOL_TITLES[user_id] # contains title in the chosen language only
        
        return [protocol_titles[num] for num in protocol_num]
    
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
                if curr_suggestions[i] and curr_suggestions[j] in self.PROTOCOL_TITLES[user_id]: 
                    suggestions.extend([curr_suggestions[i], curr_suggestions[j]])
            else:
                suggestions.extend(curr_suggestions)
            suggestions = set(suggestions)
            suggestions = list(suggestions)

        # if there's less than 4 suggestions, add random protocols (except protocol 6 & 11) w/o repetition 
        while len(suggestions) < 4: 
            p = random.choice([i for i in range(1,20) if i not in [6,11]]) 
            if (any(self.PROTOCOL_TITLES[user_id][p] not in curr_suggestions for curr_suggestions in list(self.suggestions[user_id]))
                and self.PROTOCOL_TITLES[user_id][p] not in self.recent_protocols and self.PROTOCOL_TITLES[user_id][p] not in suggestions):
                        suggestions.append(self.PROTOCOL_TITLES[user_id][p])
                        self.suggestions[user_id].extend([self.PROTOCOL_TITLES[user_id][p]])
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
