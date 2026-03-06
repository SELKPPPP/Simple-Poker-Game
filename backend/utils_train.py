
import json
import datetime
from collections import Counter
from boto3.dynamodb.conditions import Key
from config import db_game, db_rounds, db_deck, db_training_queue
from utils_agent import get_agent, reload_agent
from model_lib.poker_rules import compare_hands, evaluate_hand, get_rank_suit


def get_state(hand, remaining_redraws, p1_score, p2_score):
    score = evaluate_hand(hand)
    hand_rank = score[0]
    top_card = score[1][0] if score[1] else 0
        
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

def train_from_db():
    """
    Get training queue from DynamoDB and train the agent.
    """
    agent = get_agent()

    # Scan training queue for pending games
    pending_games = db_training_queue.scan(
        FilterExpression=Key('status').eq('pending')
    )
    items = pending_games.get('Items', [])
    
    if not items:
        return 0

    games_processed = 0
    processed_game_ids = []

    for item in items:
        gameid = int(item['gameid'])
        try:
            process_game(gameid, agent)
            
            db_training_queue.update_item(
                Key={'gameid': gameid},
                UpdateExpression="SET #s = :done",
                ExpressionAttributeNames={'#s': 'status'},
                ExpressionAttributeValues={':done': 'done'}
            )
            games_processed += 1
            processed_game_ids.append(gameid)
        except Exception as e:
            print(f"Error processing game {gameid}: {e}")

    if games_processed > 0:
        agent.save_q_table()
        reload_agent()

        # Simple logging
        log_entry = {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "gameids": processed_game_ids,
            "updateNum": games_processed
        }
        try:
            log_file = "training_log.json"
            logs = []
            try:
                with open(log_file, "r") as f:
                    logs = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logs = []
            
            logs.append(log_entry)

            with open(log_file, "w") as f:
                json.dump(logs, f, indent=4)
        except Exception as e:
            print(f"Failed to write log: {e}")
    
    return games_processed

def process_game(gameid, agent):
    """
    Every round in the game
    """
    game_info = db_game.get_item(Key={'gameid': gameid}).get('Item')
    if not game_info:
        return

    current_p1_score = 0
    current_p2_score = 0
    
    # Iterate through 3 rounds
    for r in range(1, 4):
        rid = f"{gameid}#{r}"
        
        # Get round data
        round_data = db_rounds.get_item(Key={'rid': rid}).get('Item')
        if not round_data:
            break 
            
        # Get Agent (Player 2) initial hand and decisions for the round (Deck Table)
        deck_id = f"{gameid}#{rid}#2"
        deck_data = db_deck.get_item(Key={'deckid': deck_id}).get('Item')
        
        if not deck_data:
            continue
            
        # State
        initial_hand = [int(c) for c in deck_data['hand']]
        redraw_remaining = int(deck_data['redraw_remaining'])
        state = get_state(initial_hand, redraw_remaining, current_p1_score, current_p2_score)
        
        # Action
        redraw_indices = [int(i) for i in deck_data['decision']]
        action = agent._indices_to_action(redraw_indices)
            
        # Reward
        p1_new_hand = [int(c) for c in round_data['p1_hand']]
        p2_new_hand = [int(c) for c in round_data['p2_hand']]
        
        winner = compare_hands(p1_new_hand, p2_new_hand)
        
        reward = 0
        if winner == 2: 
            reward = 1.0
            current_p2_score += 1
        elif winner == 1:
            reward = -1.0
            current_p1_score += 1
        
        # Reward Shaping: Encourage improvement in hand quality
        old_score = evaluate_hand(initial_hand)[0]
        new_score = evaluate_hand(p2_new_hand)[0]
        reward += (new_score - old_score) * 0.1
        
        # Next State
        next_rid = f"{gameid}#{r+1}"
        next_deck_id = f"{gameid}#{next_rid}#2"
        next_deck_data = db_deck.get_item(Key={'deckid': next_deck_id}).get('Item')
        
        next_state = None
        # If the game is not over yet, prepare next state
        if current_p1_score < 2 and current_p2_score < 2 and next_deck_data:
            next_hand = [int(c) for c in next_deck_data['hand']]
            next_rd = int(next_deck_data['redraw_remaining'])
            next_state = get_state(next_hand, next_rd, current_p1_score, current_p2_score)
        
        # Update Q-Table
        agent.update(state, action, reward, next_state)