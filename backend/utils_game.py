import time
import random
import os
from collections import Counter
from utils_user import boradcast, update_user_points
from config import db_user, db_counter, db_game, db_rounds, db_deck, db_training_queue
from utils_agent import get_agent
from model_lib.poker_rules import compare_hands, evaluate_hand, get_rank_suit


Q_TABLE_PATH = os.path.join("model_lib", 'q_table.pkl')

## api functions
def game_create(socketio, username, mode):
    # check param
    res = db_user.get_item(Key={"username": username})
    if "Item" not in res:
        return {"error": "Invalid user"}, 404
    user = res["Item"]
    if user["status"] == True:
        return {"error": "User busy"}, 400

    # update data
    gameid = db_counter.update_item(
        Key={"name": "gameid"},
        UpdateExpression="ADD ct :inc",
        ExpressionAttributeValues={":inc": 1},
        ReturnValues="UPDATED_NEW"
    )["Attributes"]["ct"]
    db_game.put_item(
        Item={
            "gameid": gameid,
            "mode": mode,               # {True: pvp; False:pve}
            "p1": user['username'],
            "p2": None,
            "status": 0,                # {0: awaiting player/not start; 1: round 1; 2: round 2; 3: round3; 4: done }
            "p1_score": 0,              
            "p2_score": 0               
        }
    )
    db_user.update_item(
        Key={"username": username},
        UpdateExpression="SET #s = :trueVal, ingame = :gameid",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":trueVal": True, ":gameid": gameid}
    )
    
    # return
    if not mode:
        step(socketio, gameid, 0)
    return {"gameid": gameid}, 200

def game_join(socketio, username, gameid):
    # check param
    game = db_game.get_item(Key={"gameid": gameid}).get("Item")
    user = db_user.get_item(Key={"username": username}).get("Item")
    if not game or game['mode'] == False or game['p2'] is not None:
        return {"error": "Invalid gameid"}, 404
    if not user:
        return {"error": "Invalid user"}, 404
    if user["status"] == True:
        return {"error": "User busy"}, 400
    
    # update data
    db_game.update_item(
        Key={"gameid": gameid},
        UpdateExpression="SET p2 = :username",
        ExpressionAttributeValues={":username": username}
    )
    db_user.update_item(
        Key={"username": username},
        UpdateExpression="SET #s = :trueVal, ingame = :gameid",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":trueVal": True, ":gameid": gameid}
    )

    # return
    step(socketio, gameid, game['status'])
    return {"message": "Join success"}, 200

def game_redraw(socketio, rid, player, redraw_positions):
    # check param
    round_data = db_rounds.get_item(Key={"rid": rid}).get("Item")
    if not round_data:
        return {"error": "Round not found"}, 404
    deckid_list = round_data.get(f"p{player}_deckid", [])
    
    gameid = int(rid.split('#')[0])
    game = db_game.get_item(Key={"gameid": gameid}).get("Item")
    
    if not game:
        return {"error": "Invalid gameid"}, 404
    
    # update data
    current_hand = round_data.get(f"p{player}_hand")
    current_rd = round_data.get(f"p{player}_rd")
    new_hand = card_redraw(current_hand.copy(), redraw_positions)
    new_rd = current_rd - len(redraw_positions)
    deckid = f"{gameid}#{rid}#{player}"
    deckid_list.append(deckid)

    db_deck.put_item(
        Item={
            "deckid": deckid,
            "gameid": gameid,
            "rid": rid,
            "player": player,
            "hand": current_hand,
            "redraw_remaining": current_rd,
            "score_diff": score(gameid, player),
            "decision": redraw_positions,
        }
    )
        
    db_rounds.update_item(
        Key={"rid": rid},
        UpdateExpression = (
            f"SET p{player}_hand = :new_hand, "
            f"p{player}_rd = :new_rd, "
            f"p{player}_deckid = :deckid_list"
        ),
        ExpressionAttributeValues={
            ":new_hand": new_hand,
            ":new_rd": new_rd,
            ":deckid_list": deckid_list
        }
    )

    # return
    return {
        "message": "Redraw success",
        "new_hand": new_hand,
        "remaining_redraws": new_rd
    }, 200

def game_confirm(socketio, rid, player):
    # check param
    round_data = db_rounds.get_item(Key={"rid": rid}).get("Item")
    if not round_data:
        return {"error": "Round not found"}, 404
    gameid = int(rid.split('#')[0])
    game = db_game.get_item(Key={"gameid": gameid}).get("Item")
    if not game:
        return {"error": "Invalid gameid"}, 404
    
    player = int(player)

    # update data
    db_rounds.update_item(
        Key={"rid": rid},
        UpdateExpression=f"SET p{player}_confirm = :trueVal",
        ExpressionAttributeValues={":trueVal": True}
    )

    # PVE only: call agent to decide
    if game['mode'] == False and player == 1:
        agent_round(socketio, rid)
        return {"message": "success"}, 200

    # check other player (for both PVP and PVE)
    round_data = db_rounds.get_item(Key={"rid": rid}).get("Item")
    if round_data.get("p1_confirm") and round_data.get("p2_confirm"):
        p1_hand = round_data.get("p1_hand")
        p2_hand = round_data.get("p2_hand")
        winner = card_compare(p1_hand, p2_hand)
        
        game = db_game.get_item(Key={"gameid": gameid}).get("Item")
        current_p1_score = game.get("p1_score", 0)
        current_p2_score = game.get("p2_score", 0)
        
        if winner == 1:
            new_p1_score = current_p1_score + 1
            db_game.update_item(
                Key={"gameid": gameid},
                UpdateExpression="SET p1_score = :newScore",
                ExpressionAttributeValues={":newScore": new_p1_score}
            )
        elif winner == 2:
            new_p2_score = current_p2_score + 1
            db_game.update_item(
                Key={"gameid": gameid},
                UpdateExpression="SET p2_score = :newScore",
                ExpressionAttributeValues={":newScore": new_p2_score}
            )
        
        game = db_game.get_item(Key={"gameid": gameid}).get("Item")
        boradcast(socketio, gameid, "Round finished")

        status = game.get("status", 0)
        if status in [0,1,2,3]:
            step(socketio, gameid, status)
        
        if status not in [0,1,2]:
            qlearn(gameid)
    
    # return
    return {"message": "success"}, 200

def game_track(rid, player):
    round_data = db_rounds.get_item(Key={"rid": rid}).get("Item")
    if not round_data:
        return {"error": "Next round not found"}, 404
    handdeck = round_data.get(f"p{player}_hand")
    redraw_remaining = round_data.get(f"p{player}_rd")
    gameid = int(rid.split('#')[0])
    game = db_game.get_item(Key={"gameid": gameid}).get("Item")
    p1_score = game.get("p1_score", 0)
    p2_score = game.get("p2_score", 0)

    return {
        "handdeck": handdeck,
        "redraw": redraw_remaining,
        "p1_score": p1_score,
        "p2_score": p2_score
    }, 200
    

## util functions
def step(socketio, gameid, status):
    # game status update
    if status == 2:
        game = db_game.get_item(Key={"gameid": gameid}).get("Item")
        p1_score = game.get("p1_score", 0)
        p2_score = game.get("p2_score", 0)
        if (p1_score == 2 and p2_score == 0) or (p1_score == 0 and p2_score == 2):
            new_status = 4
        else:
            new_status = 3
    else:
        new_status = status + 1
    
    db_game.update_item(
        Key={"gameid": gameid},
        UpdateExpression="SET #s = :newStatus",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":newStatus": new_status}
    )
    if new_status in [1,2,3]:
        # create round data
        rid = f'{gameid}#{new_status}'
        if status > 0:
            prev_rid = f'{gameid}#{status}'
            prev_round_data = db_rounds.get_item(Key={"rid": prev_rid}).get("Item")
            if prev_round_data:
                p1_rd = prev_round_data.get("p1_rd", 7)
                p2_rd = prev_round_data.get("p2_rd", 7)
        else:
            p1_rd = 7
            p2_rd = 7

        db_rounds.put_item(
            Item={
                "rid": rid,
                "gameid": gameid,
                "round_num": new_status,
                "p1_hand": card_draw(),
                "p2_hand": card_draw(),
                "p1_rd": p1_rd,
                "p2_rd": p2_rd,
                "p1_deckid": [],
                "p2_deckid": [],
                "p1_confirm": False,
                "p2_confirm": False,
                "status": True
            }
        )
    else:
        # Set player status to False
        game = db_game.get_item(Key={"gameid": gameid}).get("Item")
        p1_username = game.get("p1")
        p2_username = game.get("p2")
        p1_score = game.get("p1_score", 0)
        p2_score = game.get("p2_score", 0)
        
        # Adjust user points
        if p1_score == 2 and p2_score == 0:
            update_user_points(p1_username, 2)
        elif p1_score == 0 and p2_score == 2:
            if p2_username is not None:
                update_user_points(p2_username, 2)
        elif p1_score == 2 and p2_score == 1:
            update_user_points(p1_username, 1)
        elif p1_score == 1 and p2_score == 2:
            if p2_username is not None:
                update_user_points(p2_username, 1)
            
        db_user.update_item(
            Key={"username": p1_username},
            UpdateExpression="SET #s = :falseVal, ingame = :minusOne",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":falseVal": False, ":minusOne": -1}
        )
        if p2_username is not None:
            db_user.update_item(
                Key={"username": p2_username},
                UpdateExpression="SET #s = :falseVal, ingame = :minusOne",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":falseVal": False, ":minusOne": -1}
            )
            
        # Last update message
        try:
            boradcast(socketio, gameid, "Game room closed")
            socketio.close_room(str(gameid))
        except Exception:
            pass

# Draw card
def card_draw():
    deck = [i for i in range(52)]
    return random.sample(deck, 5)

def card_redraw(hand, rd):
    used = set(hand)
    for pos in rd:
        while True:
            new = random.randint(0, 51)
            if new not in used:
                used.add(new)
                hand[pos] = new
                break
    return hand

def score(gameid, player):
    game = db_game.get_item(Key={"gameid": gameid}).get("Item")
    p1_score = game.get("p1_score", 0)
    p2_score = game.get("p2_score", 0)
    if player == 1:
        return p1_score - p2_score
    else:
        return p2_score - p1_score




def card_compare(hand1, hand2):
    return compare_hands(hand1, hand2)

def agent_round(socketio, rid):
    # model make decision

    round_data = db_rounds.get_item(Key={"rid": rid}).get("Item")


    gameid = int(rid.split('#')[0])
    game = db_game.get_item(Key={"gameid": gameid}).get("Item")
    
    # Agent is Player 2
    hand = [int(c) for c in round_data.get("p2_hand")] # Ensure ints
    remaining_redraws = int(round_data.get("p2_rd"))
    p1_score = int(game.get("p1_score", 0))
    p2_score = int(game.get("p2_score", 0))
    
    state = get_state(hand, remaining_redraws, p1_score, p2_score)
    
    # Use the singleton agent
    agent = get_agent()
    
    if agent:
        action = agent.choose_action(state, training=False)
    else:
        action = 0 # Keep hand
        
    # Convert Action to Redraw Indices
    decision = agent._actions_to_indices(action)
            
    print(f"Agent (P2) State: {state} -> Action: {action} -> Redraw: {decision}")

    # Execute Action
    game_redraw(socketio, rid, 2, decision)
    game_confirm(socketio, rid, 2)

def qlearn(gameid):
    """
    Add game to training queue.
    """
   
    db_training_queue.put_item(
        Item={
            'gameid': gameid,
            'timestamp': int(time.time()),
            'status': 'pending' # pending, done
        }
    )
 



# Helper to construct state for agent 
# Same as in poker_env.py
def get_state(hand, remaining_redraws, p1_score, p2_score):
    """
    Returns:
            HandRank: Int 0-9 (High Card to Royal Flush)
            TopCard: Int 0-12 (Rank of the most significant card)
            SuitedMask: Tuple of 5 binary values at least 3 cards share same suit
            StraightMask: Tuple of 5 binary values if at least 3 cards form part of a straight
            RemainingRedraws: Int
            P1_score: Int
            P2_score: Int
    """
    # Calculate Hand Rank and Kickers
    score = evaluate_hand(hand)
    hand_rank = score[0] # 0-9
    
    top_card = score[1][0] if score[1] else 0
        
    # Calculate Suited Mask and Ranks
    suits = []
    ranks = []
    for card in hand:
        r, s = get_rank_suit(card)
        suits.append(s)
        ranks.append(r)
            
    suit_counts = Counter(suits)
    most_common = suit_counts.most_common()
        
    dominant_suit = -1
    max_count = 0
    if most_common:
        dominant_suit = most_common[0][0]
        max_count = most_common[0][1]
            
    suited_mask = []
    for s in suits:
        if s == dominant_suit and max_count >= 3:
            suited_mask.append(1)
        else:
            suited_mask.append(0)

    # Calculate Straight Mask
    straight_sets = [set(range(i, i+5)) for i in range(9)]
    straight_sets.append({12, 0, 1, 2, 3}) # Wheel: A, 2, 3, 4, 5
    
    best_straight_mask = [0] * 5
    max_straight_count = 0
    
    for s_set in straight_sets:
        unique_hits = set()
        for r in ranks:
            if r in s_set:
                unique_hits.add(r)
        
        unique_count = len(unique_hits)
        
        if unique_count > max_straight_count:
            max_straight_count = unique_count
            best_straight_mask = [1 if r in s_set else 0 for r in ranks]
    
    if max_straight_count < 3:
        best_straight_mask = [0] * 5
                
    return (hand_rank, top_card, tuple(suited_mask), tuple(best_straight_mask), remaining_redraws, p2_score, p1_score)

