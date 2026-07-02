from typing import List, Tuple
from collections import Counter


def get_rank_suit(card) -> Tuple[int, int]:
    # 0-12: Spades, 13-25: Hearts, 26-38: Diamonds, 39-51: Clubs
    # card % 13: 0=A, 1=2, ..., 12=K
    # Poker Rank: 2=0, ..., K=11, A=12
    raw_rank = card % 13
    suit = card // 13
    
    # Adjust rank so 2 is lowest and A is highest
    # 0(A) -> 12, 1(2) -> 0, 2(3) -> 1, ...
    rank = (raw_rank + 12) % 13

    # rank: 0-12 (2 to A), suit: 0-3
    return rank, suit

def evaluate_hand(hand: List[int]) -> Tuple[int, List[int]]:
    """
    Returns a tuple representing the strength of the hand.
    (HandType, HighCard1, HighCard2, ...)
    HandType:
    9: Royal Flush
    8: Straight Flush
    7: Four of a Kind
    6: Full House
    5: Flush
    4: Straight
    3: Three of a Kind
    2: Two Pair
    1: One Pair
    0: High Card

    Args:
        hand: List of 5 card integers (0-51).
            
    Returns:
        A tuple where the first element is the hand rank (0-9),
        followed by tiebreaker card ranks in descending order.
        i.e., first compare by hand rank, then by high cards.
    
    """
    ranks = []
    suits = []
    for card in hand:
        r, s = get_rank_suit(card)
        ranks.append(r)
        suits.append(s)
    
    # Sort ranks descending for easier comparison
    ranks.sort(reverse=True)
    
    # Check Flush
    is_flush = (len(set(suits)) == 1)
    
    # Check Straight
    # Handle A-5 straight (A, 5, 4, 3, 2) -> Ranks: 12, 3, 2, 1, 0
    is_straight = False
    if len(set(ranks)) == 5:
        if ranks[0] - ranks[4] == 4:
            is_straight = True
        elif ranks == [12, 3, 2, 1, 0]: # A, 5, 4, 3, 2
            is_straight = True
            # Adjust ranks for comparison (5 is high)
            ranks = [3, 2, 1, 0, -1] # -1 for low Ace representation if needed, but for comparison 5 is high
            
    # Count frequencies
    
    counts = Counter(ranks)
    sorted_counts = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    # sorted_counts is list of (rank, count), sorted by count desc, then rank desc
    # e.g., [(rank1, count1), (rank2, count2), ...]
    # if hand is [K,K,8,8,8], sorted_counts = [(6,3), (11,2)]
    
    # 9. Royal Flush
    if is_flush and is_straight and ranks[0] == 12 and ranks[4] == 8:
        return (9, ranks)
    
    # 8. Straight Flush
    if is_flush and is_straight:
        # Handle 5-high straight flush
        if ranks == [12, 3, 2, 1, 0]:
             return (8, [3, 2, 1, 0, -1]) # 5 high
        return (8, ranks)
        
    # 7. Four of a Kind
    if sorted_counts[0][1] == 4:
        return (7, [sorted_counts[0][0]] * 4 + [sorted_counts[1][0]])
        
    # 6. Full House
    if sorted_counts[0][1] == 3 and sorted_counts[1][1] == 2:
        return (6, [sorted_counts[0][0]] * 3 + [sorted_counts[1][0]] * 2)
        
    # 5. Flush
    if is_flush:
        return (5, ranks)
        
    # 4. Straight
    if is_straight:
        if ranks == [12, 3, 2, 1, 0]:
             return (4, [3, 2, 1, 0, -1])
        return (4, ranks)
        
    # 3. Three of a Kind
    if sorted_counts[0][1] == 3:
        return (3, [sorted_counts[0][0]] * 3 + [x[0] for x in sorted_counts[1:]])
        
    # 2. Two Pair
    if sorted_counts[0][1] == 2 and sorted_counts[1][1] == 2:
        return (2, [sorted_counts[0][0]] * 2 + [sorted_counts[1][0]] * 2 + [sorted_counts[2][0]])
        
    # 1. One Pair
    if sorted_counts[0][1] == 2:
        return (1, [sorted_counts[0][0]] * 2 + [x[0] for x in sorted_counts[1:]])
        
    # 0. High Card
    return (0, ranks)

def compare_hands(hand1, hand2) -> int:
    """
    Returns 1 if hand1 wins, 2 if hand2 wins, 0 if draw
    """
    score1 = evaluate_hand(hand1)
    score2 = evaluate_hand(hand2)
    
    if score1[0] > score2[0]:
        return 1
    elif score2[0] > score1[0]:
        return 2
    else:
        # Same hand type, compare tiebreakers
        for r1, r2 in zip(score1[1], score2[1]):
            if r1 > r2:
                return 1
            elif r2 > r1:
                return 2
        return 0 # Draw
