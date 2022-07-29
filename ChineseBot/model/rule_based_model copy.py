from lib2to3.pgen2.pgen import DFAState
from locale import D_FMT
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
                "model_prompt": lambda user_id, db_session, curr_session, app: self.ask_feeling(user_id), #db_session, curr_session, app
                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_opening(user_id, app, db_session)
                },
                "protocols": {"open_text": []},
            },

            "guess_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_guess_emotion(
                    user_id, app, db_session),
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
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_check_emotion(user_id, app, db_session),

                "choices": {
                    "Sad": lambda user_id, db_session, curr_session, app: self.get_sad_emotion(user_id),
                    "Angry": lambda user_id, db_session, curr_session, app: self.get_angry_emotion(user_id),
                    "Anxious/Scared": lambda user_id, db_session, curr_session, app: self.get_anxious_emotion(user_id),
                    "Happy/Content": lambda user_id, db_session, curr_session, app: self.get_happy_emotion(user_id),
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
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_specific_event(user_id, app, db_session),

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
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_event_is_recent(user_id, app, db_session),

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
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_revisit_recent(user_id, app, db_session),

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
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_revisit_distant(user_id, app, db_session),

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
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_more_questions(user_id, app, db_session),

                "choices": {
                    "Okay": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                    "I'd rather not": "project_emotion",
                },
                "protocols": {
                    "Okay": [],
                    "I'd rather not": [self.PROTOCOL_TITLES[13]],
                },
            },

            "displaying_antisocial_behaviour": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_antisocial(user_id, app, db_session),

                "choices": {
                    "yes": "project_emotion",
                    "no": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[14]],
                    "no": [self.PROTOCOL_TITLES[13]],
                },
            },

            "internal_persecutor_saviour": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_saviour(user_id, app, db_session),

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
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_victim(user_id, app, db_session),

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
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_controlling(user_id, app, db_session),

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
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_accusing(user_id, app, db_session),

                "choices": {
                "yes": "project_emotion",
                "no": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                "yes": self.INTERNAL_PERSECUTOR_PROTOCOLS,
                "no": [self.PROTOCOL_TITLES[13]],
                },
            },

            "rigid_thought": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_rigid_thought(user_id, app, db_session),

                "choices": {
                    "yes": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                    "no": "project_emotion",
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13]],
                    "no": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[19]],
                },
            },


            "personal_crisis": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_personal_crisis(user_id, app, db_session),

                "choices": {
                    "yes": "project_emotion",
                    "no": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[17]],
                    "no": [self.PROTOCOL_TITLES[13]],
                },
            },

            ################# POSITIVE EMOTION (HAPPINESS/CONTENT) #################

            "after_classification_positive": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_happy(user_id, app, db_session),

                "choices": {
                    "Okay": "suggestions",
                    "No, thank you": "ending_prompt"
                },
                "protocols": {
                    "Okay": [self.PROTOCOL_TITLES[9], self.PROTOCOL_TITLES[10], self.PROTOCOL_TITLES[11]], #change here?
                    #[self.PROTOCOL_TITLES[k] for k in self.positive_protocols],
                    "No, thank you": []
                },
            },

            ############################# ALL EMOTIONS #############################

            "project_emotion": {
               "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_project_emotion(user_id, app, db_session),

               "choices": {
                   "Continue": "suggestions",
               },
               "protocols": {
                   "Continue": [],
               },
            },


            "suggestions": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggestions(user_id, app, db_session),

                "choices": {
                     self.PROTOCOL_TITLES[k]: "trying_protocol" #self.current_protocol_ids[user_id]
                     for k in self.positive_protocols
                },
                "protocols": {
                     self.PROTOCOL_TITLES[k]: [self.PROTOCOL_TITLES[k]]
                     for k in self.positive_protocols
                },
            },

            "trying_protocol": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_trying_protocol(user_id, app, db_session),

                "choices": {"continue": "user_found_useful"},
                "protocols": {"continue": []},
            },

            "user_found_useful": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_found_useful(user_id, app, db_session),

                "choices": {
                    "I feel better": "new_protocol_better",
                    "I feel worse": "new_protocol_worse",
                },
                "protocols": {
                    "I feel better": [],
                    "I feel worse": []
                },
            },

            "new_protocol_better": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_better(user_id, app, db_session),

                "choices": {
                    "Yes (show follow-up suggestions)": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_new_protocol(
                        user_id, app
                    ),
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
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_worse(user_id, app, db_session),

                "choices": {
                    "Yes (show follow-up suggestions)": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_new_protocol(
                        user_id, app
                    ),
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
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ending(user_id, app, db_session),

                "choices": {"any": "opening_prompt"},
                "protocols": {"any": []}
                },

            "restart_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_restart_prompt(user_id),

                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_opening(user_id, app, db_session)
                },
                "protocols": {"open_text": []},
            },
        }
        self.QUESTION_KEYS = list(self.QUESTIONS.keys())

    
    
    def get_suggestions(self, user_id, app): #from all the lists of protocols collected at each step of the dialogue it puts together some and returns these as suggestions
        suggestions = []
        for curr_suggestions in list(self.suggestions[user_id]):
            if len(curr_suggestions) > 2:
                i, j = random.choices(range(0,len(curr_suggestions)), k=2)
                if curr_suggestions[i] and curr_suggestions[j] in self.PROTOCOL_TITLES: #weeds out some gibberish that im not sure why it's there
                    suggestions.extend([curr_suggestions[i], curr_suggestions[j]])
            else:
                suggestions.extend(curr_suggestions)
            suggestions = set(suggestions)
            suggestions = list(suggestions)
        while len(suggestions) < 4: #augment the suggestions if less than 4, we add random ones avoiding repetitions
            p = random.choice([i for i in range(1,20) if i not in [6,11]]) #we dont want to suggest protocol 6 or 11 at random here
            if (any(self.PROTOCOL_TITLES[p] not in curr_suggestions for curr_suggestions in list(self.suggestions[user_id]))
                and self.PROTOCOL_TITLES[p] not in self.recent_protocols and self.PROTOCOL_TITLES[p] not in suggestions):
                        suggestions.append(self.PROTOCOL_TITLES[p])
                        self.suggestions[user_id].extend([self.PROTOCOL_TITLES[p]])
        return suggestions


    def update_suggestions(self, user_id, protocols, app):

        # Check if user_id already has suggestions
        try:
            self.suggestions[user_id]
        except KeyError:
            self.suggestions[user_id] = []

        if type(protocols) != list:
            self.suggestions[user_id].append(deque([protocols]))
        else:
            self.suggestions[user_id].append(deque(protocols))

    # Takes next item in queue, or moves on to suggestions
    # if all have been checked



    def get_next_question(self, user_id):
        if self.remaining_choices[user_id] == []:
            return "project_emotion"
        else:
            selected_choice = np.random.choice(self.remaining_choices[user_id])
            self.remaining_choices[user_id].remove(selected_choice)
            return selected_choice

    def add_to_reordered_protocols(self, user_id, next_protocol):
        self.reordered_protocol_questions[user_id].append(next_protocol)

    def add_to_next_protocols(self, next_protocols):
        self.protocols_to_suggest.append(deque(next_protocols))

    # NOTE: this is not currently used, but can be integrated to support
    # positive protocol suggestions (to avoid recent protocols).
    # You would need to add it in when a user's emotion is positive
    # and they have chosen a protocol.

    def add_to_recent_protocols(self, recent_protocol):
        if len(self.recent_protocols) == self.recent_protocols.maxlen:
            # Removes oldest protocol
            self.recent_protocols.popleft()
        self.recent_protocols.append(recent_protocol)


    def determine_next_prompt_opening(self, user_id, app, db_session):
        user_response = self.user_choices[user_id]["choices_made"]["ask_feeling"]
        emotion = get_emotion(user_response, self.language[user_id])
        # if emotion == 'fear':
        #     self.guess_emotion_predictions[user_id] = 'Anxious/Scared'
        #     self.user_emotions[user_id] = 'Anxious'
        # elif emotion == 'sadness':
        #     self.guess_emotion_predictions[user_id] = 'Sad'
        #     self.user_emotions[user_id] = 'Sad'
        # elif emotion == 'anger':
        #     self.guess_emotion_predictions[user_id] = 'Angry'
        #     self.user_emotions[user_id] = 'Angry'
        # else:
        #     self.guess_emotion_predictions[user_id] = 'Happy/Content'
        #     self.user_emotions[user_id] = 'Happy'
        self.guess_emotion_predictions[user_id] = emotion
        self.user_emotions[user_id] = emotion

        return "guess_emotion"


    def get_model_prompt_guess_emotion(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        column_name = "All emotions - From what you have said I believe you are feeling {}. Is this correct?"
        my_string = self.get_sentence(column_name, user_id)
        # if len(self.recent_questions[user_id]) < 50:
        #     self.recent_questions[user_id].append(my_string)
        # else:
        #     self.recent_questions[user_id] = []
        #     self.recent_questions[user_id].append(my_string)
        question = my_string.format(self.guess_emotion_predictions[user_id].lower())    # places the emotions inside {}
        return self.split_sentence(question)

    def get_model_prompt_check_emotion(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column_name = "All emotions - I am sorry. Please select from the emotions below the one that best reflects what you are feeling:"
        my_string = self.get_sentence(column_name, user_id)
        # if len(self.recent_questions[user_id]) < 50:
        #     self.recent_questions[user_id].append(my_string)
        # else:
        #     self.recent_questions[user_id] = []
        #     self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)

    def get_sad_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "Sad"
        self.user_emotions[user_id] = "Sad"
        return "after_classification_negative"
    def get_angry_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "Angry"
        self.user_emotions[user_id] = "Angry"
        return "after_classification_negative"
    def get_anxious_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "Anxious/Scared"
        self.user_emotions[user_id] = "Anxious"
        return "after_classification_negative"
    def get_happy_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "Happy/Content"
        self.user_emotions[user_id] = "Happy"
        return "after_classification_positive"




    def determine_next_prompt_new_protocol(self, user_id, app):
        try:
            self.suggestions[user_id]
        except KeyError:
            self.suggestions[user_id] = []
        if len(self.suggestions[user_id]) > 0:
            return "suggestions"
        return "more_questions"


    def determine_positive_protocols(self, user_id, app):
        protocol_counts = {}
        total_count = 0

        for protocol in self.positive_protocols:
            count = Protocol.query.filter_by(protocol_chosen=protocol).count()
            protocol_counts[protocol] = count
            total_count += count

        # for protocol in counts:
        if total_count > 10:
            first_item = min(zip(protocol_counts.values(), protocol_counts.keys()))[1]
            del protocol_counts[first_item]

            second_item = min(zip(protocol_counts.values(), protocol_counts.keys()))[1]
            del protocol_counts[second_item]

            third_item = min(zip(protocol_counts.values(), protocol_counts.keys()))[1]
            del protocol_counts[third_item]
        else:
            # CASE: < 10 protocols undertaken in total, so randomness introduced
            # to avoid lowest 3 being recommended repeatedly.
            # Gives number of next protocol to be suggested
            first_item = np.random.choice(
                list(set(self.positive_protocols) - set(self.recent_protocols))
            )
            second_item = np.random.choice(
                list(
                    set(self.positive_protocols)
                    - set(self.recent_protocols)
                    - set([first_item])
                )
            )
            third_item = np.random.choice(
                list(
                    set(self.positive_protocols)
                    - set(self.recent_protocols)
                    - set([first_item, second_item])
                )
            )

        return [
            self.PROTOCOL_TITLES[first_item],
            self.PROTOCOL_TITLES[second_item],
            self.PROTOCOL_TITLES[third_item],
        ]

    def determine_protocols_keyword_classifiers(
        self, user_id, db_session, curr_session, app
    ):

        # We add "suggestions" first, and in the event there are any left over we use those, otherwise we divert past it.
        self.add_to_reordered_protocols(user_id, "suggestions")

        # Default case: user should review protocols 13 and 14.
        #self.add_to_next_protocols([self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[14]])
        return self.get_next_protocol_question(user_id, app)


    def update_conversation(self, user_id, new_dialogue, db_session, app):
        try:
            session_id = self.user_choices[user_id]["current_session_id"]
            curr_session = UserModelSession.query.filter_by(id=session_id).first()
            if curr_session.conversation is None:
                curr_session.conversation = "" + new_dialogue
            else:
                curr_session.conversation = curr_session.conversation + new_dialogue
            curr_session.last_updated = datetime.datetime.utcnow()
            db_session.commit()
        except KeyError:
            curr_session = UserModelSession(
                user_id=user_id,
                conversation=new_dialogue,
                last_updated=datetime.datetime.utcnow(),
            )

            db_session.add(curr_session)
            db_session.commit()
            self.user_choices[user_id]["current_session_id"] = curr_session.id


    def save_current_choice(
        self, user_id, input_type, user_choice, user_session, db_session, app
    ):
        # Set up dictionary if not set up already
        # with Session() as session:

        # try:
        #     self.user_choices[user_id]
        # except KeyError:
        #     self.user_choices[user_id] = {}

        # # Define default choice if not already set
        # try:
        #     current_choice = self.user_choices[user_id]["choices_made"]["current_choice"]
        # except KeyError:
        #     current_choice = self.QUESTION_KEYS[0]

        # try:
        #     self.user_choices[user_id]["choices_made"]
        # except KeyError:
        #     self.user_choices[user_id]["choices_made"] = {}

        # if current_choice == "ask_name":
        #     self.clear_suggestions(user_id)
        #     self.user_choices[user_id]["choices_made"] = {}
        #     self.create_new_run(user_id, db_session, user_session)

        # Save current choice
        current_choice = self.user_choices[user_id]["choices_made"]["current_choice"]
        self.user_choices[user_id]["choices_made"][current_choice] = user_choice
        curr_prompt = self.QUESTIONS[current_choice]["model_prompt"]
        
        # if callable(curr_prompt):
        #     curr_prompt = curr_prompt(user_id, db_session, user_session, app)

        # else:
        #     self.update_conversation(
        #         user_id,
        #         "Model:{} \nUser:{} \n".format(curr_prompt, user_choice),
        #         db_session,
        #         app,
        #     )

        # Case: update suggestions for next attempt by removing relevant one
        if (
            current_choice == "suggestions"
        ):

            # PRE: user_choice is a string representing a number from 1-20,
            # or the title for the corresponding protocol

            try:
                current_protocol = self.TITLE_TO_PROTOCOL[user_choice]
            except KeyError:
                current_protocol = int(user_choice)

            protocol_chosen = Protocol(
                protocol_chosen=current_protocol,
                user_id=user_id,
                session_id=user_session.id,
                run_id=self.current_run_ids[user_id],
            )
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

        # PRE: User choice is string in ["Better", "Worse"]
        elif current_choice == "user_found_useful":
            current_protocol = Protocol.query.filter_by(
                id=self.current_protocol_ids[user_id][1]
            ).first()
            current_protocol.protocol_was_useful = user_choice
            db_session.commit()

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

    def determine_next_choice(
        self, user_id, input_type, user_choice, db_session, user_session, app
    ):
        # Find relevant user info by using user_id as key in dict.
        #
        # Then using the current choice and user input, we determine what the next
        # choice is and return this as the output.

        # Some edge cases to consider based on the different types of each field:
        # May need to return list of model responses. For next protocol, may need
        # to call function if callable.

        # If we cannot find the specific choice (or if None etc.) can set user_choice
        # to "any".

        # PRE: Will be defined by save_current_choice if it did not already exist.
        # (so cannot be None)

        current_choice = self.user_choices[user_id]["choices_made"]["current_choice"]
        current_choice_for_question = self.QUESTIONS[current_choice]["choices"]
        current_protocols = self.QUESTIONS[current_choice]["protocols"]
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
                # and current_choice != "choose_persona"
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

            elif current_choice == "check_emotion":
                if user_choice == "Sad":
                    next_choice = current_choice_for_question["Sad"]
                    protocols_chosen = current_protocols["Sad"]
                elif user_choice == "Angry":
                    next_choice = current_choice_for_question["Angry"]
                    protocols_chosen = current_protocols["Angry"]
                elif user_choice == "Anxious/Scared":
                    next_choice = current_choice_for_question["Anxious/Scared"]
                    protocols_chosen = current_protocols["Anxious/Scared"]
                else:
                    next_choice = current_choice_for_question["Happy/Content"]
                    protocols_chosen = current_protocols["Happy/Content"]
            else:
                next_choice = current_choice_for_question[user_choice]
                protocols_chosen = current_protocols[user_choice]

        else:
            next_choice = current_choice_for_question["open_text"]  # for ask_feeling, it's the determine_next_opening_prompt function
            protocols_chosen = current_protocols["open_text"]       # for ask_feeling, it's []

        if callable(next_choice):
            next_choice = next_choice(user_id, db_session, user_session, app) # gets the prompt name 'guess_emotion'

        if current_choice == "guess_emotion" and user_choice.lower() == "yes":
            if self.guess_emotion_predictions[user_id] == "Sad":
                next_choice = next_choice["Sad"]
            elif self.guess_emotion_predictions[user_id] == "Angry":
                next_choice = next_choice["Angry"]
            elif self.guess_emotion_predictions[user_id] == "Anxious/Scared":
                next_choice = next_choice["Anxious/Scared"]
            else:
                next_choice = next_choice["Happy/Content"]

        if callable(protocols_chosen):
            protocols_chosen = protocols_chosen(user_id, db_session, user_session, app)

        next_prompt = self.QUESTIONS[next_choice]["model_prompt"]               # get_model_prompt_guess_emotion function
        if callable(next_prompt):
            next_prompt = next_prompt(user_id, db_session, user_session, app)   # the full prompt "i believe you are feeling {}"
        
        if (len(protocols_chosen) > 0 and current_choice != "suggestions"):
            self.update_suggestions(user_id, protocols_chosen, app)

        # Case: new suggestions being created after first protocol attempted
        if next_choice == "ask_feeling":
            self.clear_suggestions(user_id)
            self.clear_emotion_scores(user_id)
            self.create_new_run(user_id, db_session, user_session)

        if next_choice == "suggestions":
            next_choices = self.get_suggestions(user_id, app)

        else:
            next_choices = list(self.QUESTIONS[next_choice]["choices"].keys())          # [yes, no]

        self.user_choices[user_id]["choices_made"]["current_choice"] = next_choice

        return {"model_prompt": next_prompt, "choices": next_choices}
    
    
    ###################################################################
    ########## Function to set up/ initialise the chatbot #############
    ###################################################################
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

    def initialise_prev_questions(self, user_id):
        self.recent_questions[user_id] = []

    def set_language(self, user_id, language):
        '''
        Sets the language and extracts the dataset in the relevant language.
        - user_id [int]
        - language [str]: either "中文(ZH)" or "English(EN)"
        '''
        self.language[user_id] = language
        self.datasets[user_id] = pd.read_csv(f'{language}.csv', encoding='ISO-8859-1') 

    #############################################################################
    ######## Function to extract and process sentences before outputting ########
    #############################################################################
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

        # temp_list = [i + " " if i[-1] in [".", "?", "!"] else i for i in temp_list]
        # if len(temp_list) == 2:
        #     return temp_list[0], temp_list[1]
        # elif len(temp_list) == 3:
        #     return temp_list[0], temp_list[1], temp_list[2]
        # else:
        return tuple(temp_list)

    #############################################################################
    ########                    G E T  P R O M P T S                     ########
    #############################################################################
    def ask_feeling(self, user_id):
        '''
        Opening prompt: ask users how they feel.
        - user_id [int]

        Returns:
        - the opening prompt in the selected language [str]
        '''
        opening_prompt = {
            'English(EN)': "Hello, my name is Kai and I will be here to assist you today! How are you feeling today?",
            "中文(ZH)": "你好，我叫凯。我是您今天的助手！请问您今天感觉如何?"
        }
        
        return opening_prompt[self.language[user_id]]

    def get_restart_prompt(self, user_id):
        '''
        Restart prompt: ask users how they feel again.
        - user_id [int]

        Returns:
        - the restart prompt in the selected language [str]
        '''
        restart_prompt = {
            'English(EN)': "Please tell me again, how are you feeling today?",
            "中文(ZH)": "请您再告诉我您今天感觉如?"
        }
        
        return restart_prompt[self.language[user_id]]



    def get_model_prompt_project_emotion(self, user_id, app, db_session):

        prompt = "Thank you. While I have a think about which protocols would be best for you, please take your time now and try to project your current " + self.user_emotions[user_id].lower() + " emotion onto your childhood self. When you are able to do this, please press 'continue' to receive your suggestions."
        return self.split_sentence(prompt)

    def get_model_prompt_saviour(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        # data = self.datasets[user_id]
        column_name = self.user_emotions[user_id] + " - Do you believe that you should be the saviour of someone else?"
        
        question = self.get_sentence(column_name, user_id)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_victim(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        # data = self.datasets[user_id]
        column_name = self.user_emotions[user_id] + " - Do you see yourself as the victim, blaming someone else for how negative you feel?"
        
        question = self.get_sentence(column_name, user_id)
        # if len(self.recent_questions[user_id]) < 50:
        #     self.recent_questions[user_id].append(question)
        # else:
        #     self.recent_questions[user_id] = []
        #     self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_controlling(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column_name = self.user_emotions[user_id] + " - Do you feel that you are trying to control someone?"
        
        question = self.get_sentence(column_name, user_id)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_accusing(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column_name = self.user_emotions[user_id] + " - Are you always blaming and accusing yourself for when something goes wrong?"
        
        question = self.get_sentence(column_name, user_id)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_specific_event(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column_name = self.user_emotions[user_id] + " - Was this caused by a specific event/s?"
        
        question = self.get_sentence(column_name, user_id)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_event_is_recent(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column_name = self.user_emotions[user_id] + " - Was this caused by a recent or distant event (or events)?"
        
        question = self.get_sentence(column_name, user_id)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_revisit_recent(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column_name = self.user_emotions[user_id] + " - Have you recently attempted protocol 11 and found this reignited unmanageable emotions as a result of old events?"
        
        question = self.get_sentence(column_name, user_id)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_revisit_distant(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column_name = self.user_emotions[user_id] + " - Have you recently attempted protocol 6 and found this reignited unmanageable emotions as a result of old events?"
        
        question = self.get_sentence(column_name, user_id)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_more_questions(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column_name = self.user_emotions[user_id] + " - Thank you. Now I will ask some questions to understand your situation."
        
        question = self.get_sentence(column_name, user_id)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_antisocial(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column_name = self.user_emotions[user_id] + " - Have you strongly felt or expressed any of the following emotions towards someone:"
        
        question = self.get_sentence(column_name, user_id)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return [self.split_sentence(question), "Envy, jealousy, greed, hatred, mistrust, malevolence, or revengefulness?"]

    def get_model_prompt_rigid_thought(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column_name = self.user_emotions[user_id] + " - In previous conversations, have you considered other viewpoints presented?"
        
        question = self.get_sentence(column_name, user_id)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_personal_crisis(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column_name = self.user_emotions[user_id] + " - Are you undergoing a personal crisis (experiencing difficulties with loved ones e.g. falling out with friends)?"
        
        question = self.get_sentence(column_name, user_id)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_happy(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column_name = "Happy - That's Good! Let me recommend a protocol you can attempt."
        question = self.get_sentence(column_name, user_id)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_suggestions(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column_name = "All emotions - Here are my recommendations, please select the protocol that you would like to attempt"
        question = self.get_sentence(column_name, user_id)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_trying_protocol(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column_name = "All emotions - Please try to go through this protocol now. When you finish, press 'continue'"
        question = self.get_sentence(column_name, user_id)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return ["You have selected Protocol " + str(self.current_protocol_ids[user_id][0]) + ". ", self.split_sentence(question)]

    def get_model_prompt_found_useful(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column_name = "All emotions - Do you feel better or worse after having taken this protocol?"
        question = self.get_sentence(column_name, user_id)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_new_better(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column_name = "All emotions - Would you like to attempt another protocol? (Patient feels better)"
        question = self.get_sentence(column_name, user_id)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_new_worse(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column_name = "All emotions - Would you like to attempt another protocol? (Patient feels worse)"
        question = self.get_sentence(column_name, user_id)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_ending(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column_name = "All emotions - Thank you for taking part. See you soon"
        question = self.get_sentence(column_name, user_id)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return [self.split_sentence(question), "You have been disconnected. Refresh the page if you would like to start over."]



# commented out:
# def save_name(self, user_id):
    #     try:
    #         user_response = self.user_choices[user_id]["choices_made"]["ask_name"]
    #     except:  # noqa
    #         user_response = ""
    #     self.users_names[user_id] = user_response
    #     return "choose_persona"


# commented out:
# def save_name(self, user_id):
    #     try:
    #         user_response = self.user_choices[user_id]["choices_made"]["ask_name"]
    #     except:  # noqa
    #         user_response = ""
    #     self.users_names[user_id] = user_response
    #     return "choose_persona"

# replaced by get_model_prompt():
    # def get_model_prompt_check_emotion(self, user_id, app, db_session):

    #     column_name = 
    #     my_string = self.get_sentence(column_name, user_id)
        
    #     return self.split_sentence(my_string)

    # def get_model_prompt_found_useful(self, user_id, app, db_session):
    #     column_name = "All emotions - Do you feel better or worse after having taken this protocol?"
    #     question = self.get_sentence(column_name, user_id)
        
    #     return self.split_sentence(question)

    # def get_model_prompt_new_better(self, user_id, app, db_session):
    #     column_name = "All emotions - Would you like to attempt another protocol? (Patient feels better)"
    #     question = self.get_sentence(column_name, user_id)
        
    #     return self.split_sentence(question)

    # def get_model_prompt_new_worse(self, user_id, app, db_session):
    #     column_name = "All emotions - Would you like to attempt another protocol? (Patient feels worse)"
    #     question = self.get_sentence(column_name, user_id)
        
    #     return self.split_sentence(question)

    # def get_model_prompt_happy(self, user_id, app, db_session):
    #     column_name = "Happy - That's Good! Let me recommend a protocol you can attempt."
    #     question = self.get_sentence(column_name, user_id)
        
    #     return self.split_sentence(question)

    # def get_model_prompt_suggestions(self, user_id, app, db_session):
    #     column_name = "All emotions - Here are my recommendations, please select the protocol that you would like to attempt"
    #     question = self.get_sentence(column_name, user_id)
        
    #     return self.split_sentence(question)

    ####### Special ones #######
    # def get_model_prompt_rigid_thought(self, user_id, app, db_session):
    #     column_name = self.user_emotions[user_id] + " - In previous conversations, have you considered other viewpoints presented?"
    #     question = self.get_sentence(column_name, user_id)
        
    #     return self.split_sentence(question)

    # def get_model_prompt_personal_crisis(self, user_id, app, db_session):
    #     column_name = self.user_emotions[user_id] + " - Are you undergoing a personal crisis (experiencing difficulties with loved ones e.g. falling out with friends)?"
    #     question = self.get_sentence(column_name, user_id)
        
    #     return self.split_sentence(question)

    # def get_model_prompt_saviour(self, user_id, app, db_session):
    #     column_name = self.user_emotions[user_id] + " - Do you believe that you should be the saviour of someone else?"
    #     question = self.get_sentence(column_name, user_id)
        
    #     return self.split_sentence(question)

    # def get_model_prompt_victim(self, user_id, app, db_session):
    #     column_name = self.user_emotions[user_id] + " - Do you see yourself as the victim, blaming someone else for how negative you feel?"
    #     question = self.get_sentence(column_name, user_id)

    #     return self.split_sentence(question)

    # def get_model_prompt_controlling(self, user_id, app, db_session):
    #     column_name = self.user_emotions[user_id] + " - Do you feel that you are trying to control someone?"
    #     question = self.get_sentence(column_name, user_id)

    #     return self.split_sentence(question)

    # def get_model_prompt_accusing(self, user_id, app, db_session):
    #     column_name = self.user_emotions[user_id] + " - Are you always blaming and accusing yourself for when something goes wrong?"
    #     question = self.get_sentence(column_name, user_id)
        
    #     return self.split_sentence(question)

    # def get_model_prompt_specific_event(self, user_id, app, db_session):
    #     column_name = self.user_emotions[user_id] + " - Was this caused by a specific event/s?"
    #     question = self.get_sentence(column_name, user_id)
        
    #     return self.split_sentence(question)

    # def get_model_prompt_event_is_recent(self, user_id, app, db_session):
    #     column_name = self.user_emotions[user_id] + " - Was this caused by a recent or distant event (or events)?"
    #     question = self.get_sentence(column_name, user_id)
        
    #     return self.split_sentence(question)

    # def get_model_prompt_revisit_recent(self, user_id, app, db_session):
    #     column_name = self.user_emotions[user_id] + " - Have you recently attempted protocol 11 and found this reignited unmanageable emotions as a result of old events?"
    #     question = self.get_sentence(column_name, user_id)
        
    #     return self.split_sentence(question)

    # def get_model_prompt_revisit_distant(self, user_id, app, db_session):
    #     column_name = self.user_emotions[user_id] + " - Have you recently attempted protocol 6 and found this reignited unmanageable emotions as a result of old events?"
    #     question = self.get_sentence(column_name, user_id)
        
    #     return self.split_sentence(question)

    # def get_model_prompt_more_questions(self, user_id, app, db_session):
    #     column_name = self.user_emotions[user_id] + " - Thank you. Now I will ask some questions to understand your situation."
    #     question = self.get_sentence(column_name, user_id)
        
    #     return self.split_sentence(question)

    ##### took out this: cause not used???
    # def determine_positive_protocols(self, user_id, app):
    #     protocol_counts = {}
    #     total_count = 0

    #     for protocol in self.positive_protocols:
    #         count = Protocol.query.filter_by(protocol_chosen=protocol).count()
    #         protocol_counts[protocol] = count
    #         total_count += count

    #     # for protocol in counts:
    #     if total_count > 10:
    #         first_item = min(zip(protocol_counts.values(), protocol_counts.keys()))[1]
    #         del protocol_counts[first_item]

    #         second_item = min(zip(protocol_counts.values(), protocol_counts.keys()))[1]
    #         del protocol_counts[second_item]

    #         third_item = min(zip(protocol_counts.values(), protocol_counts.keys()))[1]
    #         del protocol_counts[third_item]
    #     else:
    #         # CASE: < 10 protocols undertaken in total, so randomness introduced
    #         # to avoid lowest 3 being recommended repeatedly.
    #         # Gives number of next protocol to be suggested
    #         first_item = np.random.choice(
    #             list(set(self.positive_protocols) - set(self.recent_protocols))
    #         )
    #         second_item = np.random.choice(
    #             list(
    #                 set(self.positive_protocols)
    #                 - set(self.recent_protocols)
    #                 - set([first_item])
    #             )
    #         )
    #         third_item = np.random.choice(
    #             list(
    #                 set(self.positive_protocols)
    #                 - set(self.recent_protocols)
    #                 - set([first_item, second_item])
    #             )
    #         )

    #     return [
    #         self.PROTOCOL_TITLES[first_item],
    #         self.PROTOCOL_TITLES[second_item],
    #         self.PROTOCOL_TITLES[third_item],
    #     ]

    # def add_to_next_protocols(self, next_protocols):
    #     self.protocols_to_suggest.append(deque(next_protocols))

    # def add_to_recent_protocols(self, recent_protocol):
    #     # # NOTE: this is not currently used, but can be integrated to support
    #     # # positive protocol suggestions (to avoid recent protocols).
    #     # # You would need to add it in when a user's emotion is positive
    #     # # and they have chosen a protocol.
    #     if len(self.recent_protocols) == self.recent_protocols.maxlen:
    #         # Removes oldest protocol
    #         self.recent_protocols.popleft()
    #     self.recent_protocols.append(recent_protocol)

    # def add_to_reordered_protocols(self, user_id, next_protocol):
    #     self.reordered_protocol_questions[user_id].append(next_protocol)

    # def determine_protocols_keyword_classifiers(self, user_id, db_session, curr_session, app):

    #     # We add "suggestions" first, and in the event there are any left over we use those, otherwise we divert past it.
    #     self.add_to_reordered_protocols(user_id, "suggestions")

    #     # Default case: user should review protocols 13 and 14.
    #     #self.add_to_next_protocols([self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[14]])
    #     return self.get_next_protocol_question(user_id, app)