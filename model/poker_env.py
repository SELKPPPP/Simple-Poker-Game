import random
import sys
import os
from typing import List, Tuple
from collections import Counter
from poker_rules import evaluate_hand, compare_hands, get_rank_suit


class PokerEnvironment:
    def __init__(self) -> None:
        self.deck = list(range(52))
        self.player_hand = []
        self.opponent_hand = []
        
        # Global Game State
        self.remaining_redraws = 7
        self.opponent_remaining_redraws = 7 # Opponent also has a limit
        self.player_wins = 0
        self.opponent_wins = 0
        self.round_num = 0

        # State space and action space
        self.action_space = 32  # 2^5 possible redraw combinations

        self.reset()

    def _start_round(self):
        """Deals cards for a new round."""
        random.shuffle(self.deck)
        self.player_hand = self._sort_hand(self.deck[:5])
        self.opponent_hand = self.deck[5:10]    

    def _sort_hand(self, hand: List[int]) -> List[int]:
        """
        Sort hand by rank (ascending), then by suit.
        """
        return sorted(hand, key=lambda card: get_rank_suit(card))

    def reset(self) -> Tuple[int, int, Tuple[int, ...], int, int, int]:
        """
        Reset environment to initial state for a new BO3 game.

        Returns:
            state: Initial state tuple
        """
        self.remaining_redraws = 7
        self.opponent_remaining_redraws = 7
        self.player_wins = 0
        self.opponent_wins = 0
        self.round_num = 1
        
        self._start_round()
        
        return self._get_state(self.player_hand)

    

    def _action_to_indices(self, action: int) -> List[int]:
        """
        Convert action index (0-31) to a list of indices to redraw.

        eg: action 13 (binary 01101) -> redraw indices [0, 2, 3]
        """
        redraw_indices = []
        for i in range(5):
            if (action >> i) & 1:
                redraw_indices.append(i)
        return redraw_indices
    
    def redraw(self, action: int) -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...], int, int, int], float, int, bool]:
        """
        Execute the redraw action for the current round.
        
        Args:
            action: Int 0-31
            
        Returns:
            next_state: The new state (start of next round, or terminal state).
            reward: The reward.
            result: Round result (1 win, 2 loss, 0 draw).
            done: True if BO3 game is over.
        """
        redraw_indices = self._action_to_indices(action)
        num_redraws = len(redraw_indices)
        
        # Check if action is valid (enough redraws left)
        penalty = 0.0
        if num_redraws > self.remaining_redraws:
            # Invalid action: Force "Keep Hand" (action 0)
            redraw_indices = []
            num_redraws = 0
            # Strict Penalty: Teach AI to respect the limit
            penalty = -0.5
        
        # Execute Redraw
        current_deck_idx = 10
        old_hand_score = evaluate_hand(self.player_hand)
        
        new_hand = self.player_hand.copy()
        for idx in redraw_indices:
            new_hand[idx] = self.deck[current_deck_idx]
            current_deck_idx += 1
            
        self.player_hand = self._sort_hand(new_hand)
        self.remaining_redraws -= num_redraws
        
        new_hand_score = evaluate_hand(self.player_hand)

        # Opponent's Turn
        opponent_redraw_indices = self._opponent_strategy()
        
        # Enforce opponent redraw limit
        if len(opponent_redraw_indices) > self.opponent_remaining_redraws:
            # Truncate to remaining limit
            opponent_redraw_indices = opponent_redraw_indices[:self.opponent_remaining_redraws]

        for idx in opponent_redraw_indices:
            self.opponent_hand[idx] = self.deck[current_deck_idx]
            current_deck_idx += 1
            
        self.opponent_remaining_redraws -= len(opponent_redraw_indices)
        
        # Evaluate Round Result
        result = compare_hands(self.player_hand, self.opponent_hand)
        
        win_lose_reward = 0.0
        if result == 1:
            self.player_wins += 1
            win_lose_reward = 1.0
        elif result == 2:
            self.opponent_wins += 1
            win_lose_reward = -1.0
        else:
            win_lose_reward = 0.0
            
        shaping_reward = (new_hand_score[0] - old_hand_score[0]) * 0.1
        reward = shaping_reward + win_lose_reward + penalty
        
        # Check Game Over Condition (BO3)
        # Game over if someone reaches 2 wins OR 3 rounds played
        done = False
        if self.player_wins >= 2 or self.opponent_wins >= 2 or self.round_num >= 3:
            done = True
            # Optional: Bonus for winning the match
            if self.player_wins > self.opponent_wins:
                reward += 2.0
            elif self.opponent_wins > self.player_wins:
                reward -= 2.0
        else:
            # Prepare for next round
            self.round_num += 1
            self._start_round()
            
        return self._get_state(self.player_hand), reward, result, done
    

    def _get_state(self, hand: List[int]) -> Tuple[int, int, Tuple[int, ...], Tuple[int, ...], int, int, int]:
        """
        Optimized State Representation:
        (HandRank, TopCard, SuitedMask, StraightMask, RemainingRedraws, PlayerWins, OpponentWins)
        
        This reduces state space significantly by abstracting specific card ranks into HandRank + TopKicker.
        
        Args:
            hand: List of 5 card integers (0-51).

        Returns:
            state: Tuple representing the current state.


            HandRank: Int 0-9 (High Card to Royal Flush)
            TopCard: Int 0-12 (Rank of the most significant card)
            SuitedMask: Tuple of 5 binary values at least 3 cards share same suit
            StraightMask: Tuple of 5 binary values if at least 3 cards form part of a straight
            RemainingRedraws: Int
            PlayerWins: Int
            OpponentWins: Int
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
                
        return (hand_rank, top_card, tuple(suited_mask), tuple(best_straight_mask), self.remaining_redraws, self.player_wins, self.opponent_wins)

    def get_deck(self) -> List[int]:
        return self.deck
    

    def _opponent_strategy(self):
        """
        Simple rule-based strategy for the opponent.

        Returns:
            redraw_indices: List of indices (0-4) to redraw.
        """
        hand = self.opponent_hand
        # Evaluate current hand
        score = evaluate_hand(hand)
        rank_type = score[0] # 0-9
        
        redraw_indices = []
        
        # Strategy Logic

        # Straight or better: Keep hand
        if rank_type >= 4: 
            return []
            
        elif rank_type == 3:
            # Three of a kind: Keep 3, redraw 2
            # We need to find which ones are the triplet
            # evaluate_hand returns (3, [rank_of_triplet, kicker1, kicker2])
            triplet_rank = score[1][0]
            for i, card in enumerate(hand):
                r, _ = get_rank_suit(card)
                if r != triplet_rank:
                    redraw_indices.append(i)
                    
        elif rank_type == 2:
            # Two pair: Keep 4, redraw 1
            # score[1] = [pair1_rank, pair2_rank, kicker]
            pair1 = score[1][0]
            pair2 = score[1][2]
            for i, card in enumerate(hand):
                r, _ = get_rank_suit(card)
                if r != pair1 and r != pair2:
                    redraw_indices.append(i)
                    
        elif rank_type == 1:
            # One pair: Keep 2, redraw 3
            pair_rank = score[1][0]
            for i, card in enumerate(hand):
                r, _ = get_rank_suit(card)
                if r != pair_rank:
                    redraw_indices.append(i)
                    
        else:
            # Keep top 2 cards
            # Sort hand indices by rank
            sorted_indices = sorted(range(5), key=lambda i: get_rank_suit(hand[i])[0], reverse=True)
            # Keep top 2 (indices 0 and 1 in sorted), redraw rest
            redraw_indices = sorted_indices[2:]
            
        return redraw_indices
