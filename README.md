# EE 547 - Final Project：Reinforcement Learning Card Game Agent
Repository: **ee547-project**

## Student Information
- **Name:** Wanshi Cao, Ping-Hsi Hsu, Zihan Shen
- **USC Email:** wanshica@usc.edu, pinghsi@usc.edu, zihans@usc.edu


## Contents
We organize our repository structure as shown below. We split the project into three particular pieces and let each group member work on one of the three.
To the phase of deployment, necessary files from the model section will be shifted toward the backend before it's deployed on the EC2 instance.
Frontend is deployed on S3 bucked separately.
```text
EE641_PROJECT/
├── frontend/
|   ├── card.py
|   ├── index.html
|   └── script.js
├── backend/
│   ├── model_lib/
|   |   ├── agent.py
|   |   ├── poker_rules.py
|   |   └── q_table.pkl
|   ├── app.py
|   ├── config.py
|   ├── utils_agent.py
|   ├── utils_game.py
|   ├── utils_train.py
|   └── utils_user.py
├── model/
|   ├── agent.py
|   ├── poker_env.py
|   ├── poker_rules.py
|   ├── trian.py
|   ├── q_table.pkl
|   └── trian_results.png
├── ee547_project.pem
├── deploy-db.yaml
├── makefile
├── manifest.md
├── README.md
├── requirements.txt
├── Technical Demo.pdf
├── video.mp4
└── .gitignore
```

## Database setup
To initialize the database, run the followwing command. Please make sure you are having the proper AWS access and setup.
```python
make deploy-db
```

To remove it run
```python
make delete-db
```

## Lambda function setup
To initialize the lambda function, run the followwing command. Please make sure you are having the proper AWS access and setup.
```python
make deploy-lambda
```

To remove it run
```python
make delete-lambda
```

## S3 frontend bucket setup
To initialize the S3 bucket, run the followwing command. Please make sure you are having the proper AWS access and setup.
```python
make deploy-frontend
```

To remove it run
```python
make delete-frontend
```

Then upload the frontend file to the bucket through
```python
aws s3 sync ./frontend s3://ee547-poker-game-frontend-group05
```

Website of the game 

http://ee547-poker-game-frontend-group05.s3-website-us-west-1.amazonaws.com


## Environment setup
Run the following command to setup environment locally.
```python
pip install -r requirements.txt
```

## Local test
To run the application locally, both the frontend and backend must be started separately. Follow the steps below.
1. Start the Frontend (Static File Server)

From the frontend folder which containing index.html, run:
```python
cd frontend
python -m http.server 8000
```
This will host the frontend at:
http://127.0.0.1:8000/index.html

2. Update Frontend Configuration for Local Testing

Before testing locally, make sure the frontend points to your local backend.

Update the following constants in script.js:
```javascript
// Local backend API endpoint
const API_BASE = "http://127.0.0.1:5001";

// Local Socket.IO endpoint
const socket = io("http://127.0.0.1:5001");
```
This ensures that all REST API calls and WebSocket events connect to your local Flask server running on port 5001.

3. Start the Backend Server


In another terminal window, start the backend (Flask / Flask-SocketIO):
```python
cd backend
python app.py
```
Ensure your backend is configured to run on port 5001.

4. Open the Application

Visit:
```arduino
http://127.0.0.1:8000/index.html
```

## AWS deployment
Open your AWS EC2 terminal, and log in with SSH. Then run the following to start backend:
```python
git clone https://github.com/HunterShen523/ee547-project
cd ee547-project
pip install -r requirements.txt
cd backend
python app.py
```
Your backend will now be running at:
```arduino
http://50.18.67.156:5001
```
Then visit the following url you can play the game
```arduino
http://ee547-poker-game-frontend-group05.s3-website-us-west-1.amazonaws.com
```
Make sure port 5001 is open in your AWS Security Group.
