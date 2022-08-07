### SAT chatbot web app

#### Notes: 

1) Before running the code in this folder, please obtain the files 'RoBERTa_emotion_best.pt' and 'T5_empathy_best.pt' by running the Jupyter notebooks 'emotion classifier - RoBERTa fine-tuned on Emotion + our data.ipynb' and 'empathy classifier - T5 finetuned on our data.ipynb' in the NLP models folder

2) You may need to change the file paths in 'classifiers.py' and 'rule_based_model.py' to your local paths when running locally

3) This chatbot uses the react-chatbot-kit library: https://fredrikoseberg.github.io/react-chatbot-kit-docs/


#### To run the code in this folder locally, after cloning open a terminal window and do:

$ pip3 install virtualenv

$ virtualenv ./SATbot

$ cd ./SATbot

$ source bin/activate

$ cd ./model

$ python3 -m pip install -r requirements.txt

$ set FLASK_APP=flask_backend_with_aws

$ python3 -m flask db init

$ python3 -m flask db migrate -m "testDB table"

$ python3 -m flask db upgrade

$ nano .env   ---->  add DATABASE_URL="sqlite:////YOUR LOCAL PATH TO THE app.db FILE" to the .env file, save and exit

$ python3 -m flask run


#### To launch the front end, open another terminal tab and do:

$ cd ./SATbot/view

$ npm i

$ 3
