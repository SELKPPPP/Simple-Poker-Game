from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room

from config import db_game, db_rounds
from utils_user import user_auth, user_register, user_login, user_pt, boradcast
from utils_game import game_create, game_join, game_redraw, game_confirm, game_track
from utils_agent import get_agent
from utils_train import train_from_db


# Update Flask to serve static files from the frontend folder
# app = Flask(__name__, static_folder='../frontend', static_url_path='')
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize Agent on Startup
get_agent()

# Serve the frontend index.html at the root URL
# @app.route('/')
# def index():
#     return app.send_static_file('index.html')

# APIs
# User register
@app.route("/user/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400
    result, status_code = user_register(username, password)
    return jsonify(result), status_code

# User log in
@app.route("/user/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400
    result, status_code = user_login(username, password)
    return jsonify(result), status_code

# Check use current points
@app.route("/user/points", methods=["GET"])
def get_points():
    print(request.headers)
    username = user_auth(request)
    if username is None:
        return jsonify({"error": "Validation failed"}), 401
    
    result, status_code = user_pt(username)
    return jsonify(result), status_code

# Create gameroom
@app.route("/game/creategame", methods=["GET"])
def create_game():
    print(request.headers)
    username = user_auth(request)
    if username is None:
        return jsonify({"error": "Validation failed"}), 401
    
    gamemode = request.headers.get("mode")      # True for pvp, False for pve
    gamemode = True if gamemode == 'True' else False

    result, status_code = game_create(socketio, username, gamemode)
    return jsonify(result), status_code

# Join game
@app.route("/game/joingame", methods=["GET"])
def join_game():
    username = user_auth(request)
    if username is None:
        return jsonify({"error": "Validation failed"}), 401
    
    gameid = request.headers.get("gameid")

    if not gameid:
        return jsonify({"error": "Missing gameid"}), 400
    result, status_code = game_join(socketio, username, int(gameid))
    if status_code == 200:
        boradcast(socketio, int(gameid), "Player joined game")
    return jsonify(result), status_code

# Redraw cards
@app.route("/game/redraw", methods=["POST"])
def redraw():
    username = user_auth(request)
    if username is None:
        return jsonify({"error": "Validation failed"}), 401
    
    data = request.json
    rid = data.get("rid")
    player = data.get("player")  # 1 or 2
    redraw_index = data.get("redraw_positions", [])

    if not rid or not player:
        return jsonify({"error": "Missing rid or player"}), 400
    result, status_code = game_redraw(socketio, rid, player, redraw_index)
    if status_code == 200:
        gameid = int(rid.split('#')[0])
        boradcast(socketio, gameid, f"Player {player} redrew cards")
    return jsonify(result), status_code

# Confirm / ready for next round
@app.route("/game/confirm", methods=["POST"])
def confirm():
    username = user_auth(request)
    if username is None:
        return jsonify({"error": "Validation failed"}), 401
    
    data = request.json
    rid = data.get("rid")
    player = data.get("player")  # 1 or 2

    if not rid or not player:
        return jsonify({"error": "Missing rid or player"}), 400
    result, status_code = game_confirm(socketio, rid, player)

    # Only call game_confirm here; broadcasting is handled inside that function
    #if status_code == 200:
    #    gameid = int(rid.split('#')[0])
    #    boradcast(socketio, gameid, f"Player {player} confirmed")

    return jsonify(result), status_code

# Check - check status with server about whether next round start
@app.route("/game/track", methods=["POST"])
def track():
    username = user_auth(request)
    if username is None:
        return jsonify({"error": "Validation failed"}), 401
    
    data = request.json
    rid = data.get("rid")
    player = data.get("player")  # 1 or 2

    if not rid or not player:
        return jsonify({"error": "Missing rid or player"}), 400
    result, status_code = game_track(rid, player)
    return jsonify(result), status_code

# Broad casting
@socketio.on("listen")
def listen(data):
    username = user_auth(request)
    if username is None:
        return jsonify({"error": "Validation failed"}), 401
    
    gameid = str(data.get("gameid"))
    playerid = data.get("player_id")
    socketid = request.sid

    join_room(gameid)
    print(f"{socketid} joined game room: {gameid}")
    emit("system", {"join": playerid}, room=gameid)
    boradcast(socketio, int(gameid), f"Player {playerid} joined room")

# Manual Training Trigger
@app.route("/model/train", methods=["POST"])
def train_model():
    try:
        games_processed = train_from_db()
        return jsonify({"games_processed": games_processed}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    



if __name__ == "__main__":
    # app.run(debug=True)
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
