## SAT Chatbot Web App
This directory contains both the frontend and backend of the SATbot web app, currently available on: http://52.207.175.233 

### Notes: 

1) Before running the code in this folder, you will require an emotion classifier. <br>
(i) Please save your classifier in the `model` directory (eg. satbot-app/model/emotion-classifier/) <br>
(ii) Change `EMOTION_CLASSIFIER_MODEL` and `EMOTION_CLASSIFIER_TOKENIZER` in utils.py to the name of your classifier folder (eg: 'emotion-classifier')

2) This chatbot uses the react-chatbot-kit library: https://fredrikoseberg.github.io/react-chatbot-kit-docs/

3) This chatbot builds upon Lisa Alazraki's chatbot, see https://github.com/LisaAlaz/SATbot 


## Project Setup (run locally)
### I. Clone this repository: <br>
    ```
    git clone https://github.com/alicialawjy/multilingual-psychotherapy-chatbot.git
    ```

### II. Start Up Backend
1. cd into backend directory <br>
    ```
    cd multilingual-psychotherapy-chatbot/satbot-app/model
    ```

2. set up virtual environment
    ```
    pip3 install virtualenv
    virtualenv ./backendenv
    source backendenv/bin/activate
    python3 -m pip install -r requirements.txt
    ```

3. set up database
    ```
    set FLASK_APP=flask_backend_with_aws
    python3 -m flask db init
    python3 -m flask db migrate -m "testDB table"
    python3 -m flask db upgrade
    nano .env   ---->  add DATABASE_URL="sqlite:////YOUR LOCAL PATH TO THE app.db FILE" to the .env file, save and exit
    ```

4. run backend <br>
    ```
    python3 -m flask run
    ```


### III. Start Up Frontend
1. Open another terminal

2. cd into frontend terminal
    ```
    cd multilingual-psychotherapy-chatbot/satbot-app/view
    ```

3. install node_modules 
    ```
    npm i
    ```

4. run frontend
    ```
    npm run start
    ```
