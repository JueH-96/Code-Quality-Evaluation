import sys

# Set a higher recursion limit for potentially deep recursive calls
# Max recursion depth is K (max 12), but typical contest platforms might have smaller default
# and general python programs might not like too much recursion. 2000 is safe.
sys.setrecursionlimit(2000) 

# Memoization dictionary: (t_mask, a_mask, table_mask, is_takahashi_turn) -> bool (True if current player wins)
memo = {}

# Mappings for card values to indices and vice versa
# These will be populated once from input and used globally by the can_win function.
card_to_idx = {}
idx_to_card = [] # This list will be `all_cards` after sorting unique values
K = 0            # Total number of unique cards, max 12

def can_win(t_mask, a_mask, table_mask, is_takahashi_turn):
    """
    Determines if the current player can win from the given game state.

    Args:
        t_mask (int): Bitmask representing Takahashi's hand.
        a_mask (int): Bitmask representing Aoki's hand.
        table_mask (int): Bitmask representing cards on the table.
        is_takahashi_turn (bool): True if it's Takahashi's turn, False if Aoki's.

    Returns:
        bool: True if the current player wins, False otherwise.
    """
    
    state = (t_mask, a_mask, table_mask, is_takahashi_turn)
    if state in memo:
        return memo[state]

    current_player_hand_mask = t_mask if is_takahashi_turn else a_mask

    # Base case: If current player has no cards, they cannot make a move and lose.
    if current_player_hand_mask == 0:
        memo[state] = False
        return False

    # Iterate through all cards the current player can choose to play
    for c_idx in range(K):
        # Check if card with index c_idx is in the current player's hand
        if (current_player_hand_mask >> c_idx) & 1: 
            card_value_played = idx_to_card[c_idx]

            # Phase 1: Player puts card c_idx on the table
            new_current_player_hand_mask = current_player_hand_mask ^ (1 << c_idx) # Remove from hand
            temp_table_mask = table_mask | (1 << c_idx)                             # Add to table

            # Flag to track if playing card_value_played leads to a winning state
            # for the current player (by forcing the next player into a losing state).
            found_winning_Y_choice_for_X = False

            # Phase 2, Option A: The player chooses NOT to take any card from the table.
            # Determine the next player's masks
            if is_takahashi_turn:
                next_t_mask_no_take = new_current_player_hand_mask
                next_a_mask_no_take = a_mask
            else:
                next_t_mask_no_take = t_mask
                next_a_mask_no_take = new_current_player_hand_mask
            
            # If the opponent loses from this state, then this specific move (playing X, no take)
            # is a winning move for the current player.
            if not can_win(next_t_mask_no_take, next_a_mask_no_take, temp_table_mask, not is_takahashi_turn):
                found_winning_Y_choice_for_X = True

            # Phase 2, Option B: The player chooses to take a card Y from the table.
            # This option is only considered if Option A didn't already guarantee a win for this X.
            if not found_winning_Y_choice_for_X: 
                for y_idx in range(K):
                    # Check if card y_idx is currently on the table
                    if (temp_table_mask >> y_idx) & 1: 
                        card_value_taken = idx_to_card[y_idx]
                        # A card Y can only be taken if its value is less than the played card X
                        if card_value_taken < card_value_played:
                            # Player takes card y_idx into hand
                            candidate_new_current_player_hand_mask = new_current_player_hand_mask | (1 << y_idx) # Add to hand
                            candidate_new_table_mask = temp_table_mask ^ (1 << y_idx)                            # Remove from table

                            # Determine the next player's masks after taking Y
                            if is_takahashi_turn:
                                next_t_mask_take_Y = candidate_new_current_player_hand_mask
                                next_a_mask_take_Y = a_mask
                            else:
                                next_t_mask_take_Y = t_mask
                                next_a_mask_take_Y = candidate_new_current_player_hand_mask
                            
                            # If the opponent loses from this state, then this move (playing X, taking Y)
                            # is a winning move for the current player.
                            if not can_win(next_t_mask_take_Y, next_a_mask_take_Y, candidate_new_table_mask, not is_takahashi_turn):
                                found_winning_Y_choice_for_X = True
                                break # Found a winning choice for Y for this X, no need to check other Ys

            # If by playing card X (and optimally choosing to take a Y or not), 
            # the current player can force the opponent into a losing position,
            # then the current state is a winning state for the current player.
            if found_winning_Y_choice_for_X:
                memo[state] = True
                return True

    # If, after trying all possible cards X to play from hand, none of them lead to a winning state
    # (i.e., all moves result in the opponent winning), then the current state is a losing state.
    memo[state] = False
    return False

def solve():
    global card_to_idx, idx_to_card, K

    # Read input N, M, L
    N, M, L = map(int, sys.stdin.readline().split())
    # Read card lists A, B, C
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    C = list(map(int, sys.stdin.readline().split()))

    # Collect all unique card values and map them to compact indices
    all_cards_values = sorted(list(set(A + B + C)))
    K = len(all_cards_values) # K will be <= N+M+L <= 12

    idx_to_card = all_cards_values # The list itself serves as the index-to-value mapping
    card_to_idx = {card: i for i, card in enumerate(all_cards_values)} # Value-to-index mapping

    # Initialize bitmasks for the starting game state
    initial_t_mask = 0
    for card_val in A:
        initial_t_mask |= (1 << card_to_idx[card_val])

    initial_a_mask = 0
    for card_val in B:
        initial_a_mask |= (1 << card_to_idx[card_val])

    initial_table_mask = 0
    for card_val in C:
        initial_table_mask |= (1 << card_to_idx[card_val])

    # Determine the winner starting with Takahashi's turn
    if can_win(initial_t_mask, initial_a_mask, initial_table_mask, True):
        print("Takahashi")
    else:
        print("Aoki")

# Call the solve function to run the program
solve()