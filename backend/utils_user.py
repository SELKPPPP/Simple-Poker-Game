import jwt
import time
import bcrypt
from decimal import Decimal

from jwt import InvalidTokenError
from config import JWT_SECRET, db_user, db_game, db_rounds

def user_auth(req):
    try:
        token = None
        # BoradCast Usage
        token = req.args.get("token")
        # Other API usage
        if not token:
            auth_header = req.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.replace("Bearer ", "").strip()
        if not token:
            return None
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        username = payload.get("username")
        if not username:
            return None
        res = db_user.get_item(Key={"username": username})
        if "Item" not in res:
            return None
        return username
    except InvalidTokenError:
        return None
    except Exception:
        return None
    
def user_register(username, password):
    # check avaliability
    res = db_user.get_item(Key={"username": username})
    if "Item" in res:
        return {"error": "Username already exists"}, 400
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    # insert and return
    db_user.put_item(
        Item={
            "username": username,
            "passwordHash": hashed,
            "status": False,    # {False = avaliable, True = in game}
            "points": 1000,
            "ingame": -1       # {-1 = not in game, gameid = in game}
        }
    )
    return {"message": "Register success"}, 200

def user_login(username, password):
    # check identity
    res = db_user.get_item(Key={"username": username})
    if "Item" not in res:
        return {"error": "Invalid user"}, 404
    user = res["Item"]
    if not bcrypt.checkpw(password.encode(), user["passwordHash"].encode()):
        return {"error": "Invalid password"}, 400
    
    # generate Jtoken
    token = jwt.encode(
        {"username": username, "exp": int(time.time()) + 60 * 60},
        JWT_SECRET,
        algorithm="HS256"
    )
    ingame = user.get("ingame", -1)
    return {"token": token, "ingame": ingame}, 200

def user_pt(username):
    # check avaliability
    res = db_user.get_item(Key={"username": username})
    if "Item" not in res:
        return {"error": "Invalid user"}, 404
    # return value
    return {"points": res["Item"]["points"]}, 200

def update_user_points(username, points_change):
    # update points
    db_user.update_item(
        Key={"username": username},
        UpdateExpression="ADD points :pointsChange",
        ExpressionAttributeValues={":pointsChange": points_change}
    )

def normalize_hand(raw_hand):
    """
    Convert DynamoDB hand data into a clean Python list.

    Example:
        [ {'N': Decimal('19')}, {'N': Decimal('49')} ]  ->  [19, 49]

    This function normalizes different DynamoDB number formats
    (dicts, Decimals, etc.) into plain Python numbers.
    """
    if raw_hand is None or raw_hand == "NA":
        return []

    result = []
    for card in raw_hand:
        # Case 1: DynamoDB format like {'N': Decimal('19')}
        if isinstance(card, dict) and "N" in card:
            val = card["N"]
            if isinstance(val, Decimal):
                val = int(val) if val % 1 == 0 else float(val)
            else:
                val = int(val)
            result.append(val)

        # Case 2: Direct Decimal value
        elif isinstance(card, Decimal):
            result.append(int(card) if card % 1 == 0 else float(card))

        # Case 3: Any other value — append as-is）
        else:
            result.append(card)

    return result

def boradcast(socketio, gameid, msg):
    # Send message to game room users
    game = db_game.get_item(Key={"gameid": gameid}).get("Item")
    if not game:
        return
    status = game.get('status', 0)
    rid = f'{gameid}#{status}'
    round_data = db_rounds.get_item(Key={"rid": rid}).get("Item")

    message =  {
        'msg': msg,
        'round': int(status),
        'p1': game.get('p1', "NA"),
        'p1_hand': normalize_hand(round_data.get('p1_hand', "NA") if round_data else "NA"),
        'p1_rd': int(round_data.get('p1_rd', 0)) if round_data else "NA",
        'p1_score': int(game.get('p1_score', 0)),
        'p1_confirm': round_data.get('p1_confirm', "NA") if round_data else "NA",
        'p2': game.get('p2', "NA"),
        'p2_hand': normalize_hand(round_data.get('p2_hand', "NA") if round_data else "NA"),
        'p2_rd': int(round_data.get('p2_rd', 0)) if round_data else "NA",
        'p2_score': int(game.get('p2_score', 0)),
        'p2_confirm': round_data.get('p2_confirm', "NA") if round_data else "NA",
    }
    socketio.emit("system", message, room=str(gameid))
    #print("DEBUG broadcast message:", message)

