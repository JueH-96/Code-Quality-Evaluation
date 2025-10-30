import sys

# It is a good practice to increase the recursion limit for recursive solutions
# in competitive programming, although the default limit is likely sufficient
# for this problem's constraints (max depth is N+M+L <= 12).
sys.setrecursionlimit(20000)

# Memoization table to store results of subproblems.
# The key is a tuple representing the game state:
# (takahashi_hand, aoki_hand, table_cards, is_takahashi_turn)
memo = {}

def can_win(t_hand, a_hand, table, is_takahashi_turn):
    """
    Determines if the current player can win from the given game state
    using recursion with memoization. A state is winning if there exists
    a move to a losing state for the opponent.

    Args:
        t_hand: A sorted tuple of cards in Takahashi's hand.
        a_hand: A sorted tuple of cards in Aoki's hand.
        table: A sorted tuple of cards on the table.
        is_takahashi_turn: True if it's Takahashi's turn, False for Aoki.

    Returns:
        True if the current player has a winning strategy, False otherwise.
    """
    state = (t_hand, a_hand, table, is_takahashi_turn)
    if state in memo:
        return memo[state]

    if is_takahashi_turn:
        current_hand = t_hand
        if not current_hand:
            # Takahashi has no cards to play, so he loses.
            memo[state] = False
            return False
        
        # Iterate through all cards Takahashi can play.
        for i in range(len(current_hand)):
            card_to_play = current_hand[i]
            
            # Form hand and table after playing a card.
            next_hand_part = current_hand[:i] + current_hand[i+1:]
            table_with_played_card = sorted(list(table) + [card_to_play])

            # Move Option 1: Play a card and DON'T take one back.
            # If this move forces Aoki into a losing position, it's a win for Takahashi.
            if not can_win(next_hand_part, a_hand, tuple(table_with_played_card), False):
                memo[state] = True
                return True
            
            # Move Option 2: Play a card and take a smaller card back.
            # Iterate through all cards on the new table that can be taken.
            for j in range(len(table_with_played_card)):
                card_to_take = table_with_played_card[j]
                
                if card_to_take < card_to_play:
                    next_t_hand = tuple(sorted(list(next_hand_part) + [card_to_take]))
                    
                    next_table_list = list(table_with_played_card)
                    next_table_list.pop(j)
                    next_table = tuple(next_table_list)
                    
                    if not can_win(next_t_hand, a_hand, next_table, False):
                        memo[state] = True
                        return True
        
        # If no move leads to a win, this is a losing state.
        memo[state] = False
        return False
    else:  # Aoki's turn
        current_hand = a_hand
        if not current_hand:
            # Aoki has no cards to play, so he loses.
            memo[state] = False
            return False
        
        # Symmetrical logic for Aoki's turn.
        for i in range(len(current_hand)):
            card_to_play = current_hand[i]
            
            next_hand_part = current_hand[:i] + current_hand[i+1:]
            table_with_played_card = sorted(list(table) + [card_to_play])

            # Move Option 1: Don't take a card back.
            if not can_win(t_hand, next_hand_part, tuple(table_with_played_card), True):
                memo[state] = True
                return True
            
            # Move Option 2: Take a smaller card back.
            for j in range(len(table_with_played_card)):
                card_to_take = table_with_played_card[j]
                
                if card_to_take < card_to_play:
                    next_a_hand = tuple(sorted(list(next_hand_part) + [card_to_take]))
                    
                    next_table_list = list(table_with_played_card)
                    next_table_list.pop(j)
                    next_table = tuple(next_table_list)
                    
                    if not can_win(t_hand, next_a_hand, next_table, True):
                        memo[state] = True
                        return True
        
        # If no winning move is found, this is a losing state for Aoki.
        memo[state] = False
        return False

def solve():
    """
    Reads input, runs the game solver, and prints the winner.
    """
    try:
        N, M, L = map(int, sys.stdin.readline().split())
        A = list(map(int, sys.stdin.readline().split()))
        B = list(map(int, sys.stdin.readline().split()))
        C = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        return

    # Initial state must use sorted tuples for canonical representation.
    initial_t_hand = tuple(sorted(A))
    initial_a_hand = tuple(sorted(B))
    initial_table = tuple(sorted(C))
    
    # Takahashi starts. Determine if he can win from the initial state.
    if can_win(initial_t_hand, initial_a_hand, initial_table, True):
        print("Takahashi")
    else:
        print("Aoki")

solve()