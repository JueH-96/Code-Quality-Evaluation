# YOUR CODE HERE
import sys

# Setting a slightly higher recursion depth just in case, although likely not necessary for K=12
# Max recursion depth can be an issue in competitive programming.
# Using try-except block for safety.
try:
    # Increase recursion depth for potentially deep game states.
    # The maximum depth could be related to the total number of cards, N+M+L.
    # With N+M+L <= 12, the default depth of 1000 in Python should typically be sufficient, 
    # but increasing it provides a safety margin.
    sys.setrecursionlimit(2000) 
except Exception as e: 
    # In case setting recursion depth fails (e.g., due to OS limits or environment restrictions), 
    # the program will proceed with the default depth.
    # print(f"Could not set recursion depth: {e}", file=sys.stderr) # Optional: uncomment for debugging messages
    pass 

# Memoization cache (dictionary) to store results of computed game states.
# Keys are tuples representing the game state, values are booleans indicating if the current player can win.
memo = {}

def solve():
    """
    Reads input parameters N, M, L and the card lists A, B, C.
    Initializes the game state with sorted card lists represented as tuples.
    Calls the recursive function `can_win` to determine the winner assuming optimal play.
    Prints the name of the winner ("Takahashi" or "Aoki").
    """
    # Read N, M, L from the first line of input
    N, M, L = map(int, sys.stdin.readline().split())
    # Read Takahashi's initial hand (A)
    A = list(map(int, sys.stdin.readline().split()))
    # Read Aoki's initial hand (B)
    B = list(map(int, sys.stdin.readline().split()))
    # Read initial cards on the table (C)
    C = list(map(int, sys.stdin.readline().split()))

    # Sort initial hands and table cards. Sorting allows using tuples as canonical state representations.
    # Tuples are immutable and hashable, suitable for keys in the memoization dictionary.
    A.sort()
    B.sort()
    C.sort()

    # Convert the sorted lists to tuples for use in game state representation
    initial_H_T = tuple(A) # Takahashi's hand
    initial_H_A = tuple(B) # Aoki's hand
    initial_T = tuple(C)   # Cards on the table

    # Determine the winner by calling the recursive function starting with Takahashi's turn.
    # If `can_win` returns True, Takahashi has a winning strategy. Otherwise, Aoki wins.
    if can_win(initial_H_T, initial_H_A, initial_T, 'Takahashi'):
        print("Takahashi")
    else:
        print("Aoki")

def can_win(H_T, H_A, T, player):
    """
    Recursive function with memoization to determine if the current player can win from the given state.
    The game state is defined by (Takahashi's hand H_T, Aoki's hand H_A, Table cards T, current player).
    All card collections (H_T, H_A, T) are represented as sorted tuples.

    Args:
        H_T (tuple): Sorted tuple of cards in Takahashi's hand.
        H_A (tuple): Sorted tuple of cards in Aoki's hand.
        T (tuple): Sorted tuple of cards on the table.
        player (str): The current player ('Takahashi' or 'Aoki').

    Returns:
        bool: True if the current `player` can force a win from this state, False otherwise.
    """
    # Create a unique key for the current game state for memoization.
    # The key includes both players' hands, the table cards, and whose turn it is.
    state_key = (H_T, H_A, T, player)
    
    # Check if the result for this state is already computed and stored in the memo cache.
    if state_key in memo:
        return memo[state_key]

    # Determine the current player's hand based on the `player` argument.
    current_hand = H_T if player == 'Takahashi' else H_A
    
    # Base case: If the current player has an empty hand, they cannot make a move and thus lose.
    if not current_hand: 
        # Store the result (False for losing state) in the memo cache.
        memo[state_key] = False
        return False

    # Explore all possible moves for the current player.
    # Iterate through each card in the hand using its index. This approach correctly handles duplicate card values.
    for i in range(len(current_hand)):
        card_to_play = current_hand[i]
        
        # --- Calculate the state after playing the card `card_to_play` ---
        
        # Create the player's new hand tuple after removing the card at index i.
        new_hand_list = list(current_hand)
        del new_hand_list[i]
        new_hand_tuple = tuple(new_hand_list)
        
        # Create the new table list after adding the played card.
        new_T_list = list(T)
        new_T_list.append(card_to_play)
        new_T_list.sort() # Keep the table cards sorted to maintain a canonical state representation.
        
        # Identify cards currently on the table (including the one just played) 
        # that have a value strictly smaller than the played card.
        possible_takes_values = [] 
        for card_on_table in new_T_list:
             if card_on_table < card_to_play:
                 possible_takes_values.append(card_on_table)
        
        # --- Evaluate Move Option 0: Player chooses *not* to take any card from the table ---
        
        # Determine the components of the next game state for this option (Option 0).
        if player == 'Takahashi':
            next_H_T = new_hand_tuple # Takahashi's hand after playing card
            next_H_A = H_A            # Aoki's hand remains unchanged
            next_player = 'Aoki'      # Turn passes to Aoki
        else: # player == 'Aoki'
            next_H_T = H_T            # Takahashi's hand remains unchanged
            next_H_A = new_hand_tuple # Aoki's hand after playing card
            next_player = 'Takahashi' # Turn passes to Takahashi
        
        next_T = tuple(new_T_list) # The table state includes the card just played.

        # Make the recursive call for the opponent's turn from the resulting state.
        # If `can_win` returns False for the opponent, it means the current player has found a winning move.
        if not can_win(next_H_T, next_H_A, next_T, next_player):
            # Cache the result: the current state is a winning state.
            memo[state_key] = True 
            return True # Found a winning move, no need to explore further moves from this state.

        # --- Evaluate Move Options 1..k: Player takes one card `card_to_take` from the table ---
        
        # Iterate through the *unique* values of cards available to take.
        # This optimization avoids redundant recursive calls if multiple cards with the same value can be taken.
        # The list removal logic (`final_T_list.remove(card_to_take)`) correctly handles removing only one instance.
        unique_values_to_take = sorted(list(set(possible_takes_values)))

        for card_to_take in unique_values_to_take:
            # Calculate the state resulting from taking `card_to_take`.
            
            # Final hand: add the taken card to the player's hand (which already had card_to_play removed).
            final_hand_list = list(new_hand_tuple)
            final_hand_list.append(card_to_take)
            final_hand_list.sort() # Keep the hand sorted for canonical representation.
            final_hand_tuple = tuple(final_hand_list)
            
            # Final table: remove one instance of the taken card from the table list.
            final_T_list = list(new_T_list) # Start from the table state after card_to_play was added.
            # `list.remove(value)` removes the first occurrence of `value`. This correctly handles duplicate cards.
            final_T_list.remove(card_to_take) 
            final_T_tuple = tuple(final_T_list) # Convert back to tuple for state representation.

            # Determine the components of the next game state for this option (Option k).
            # The hands change based on who the current player is.
            if player == 'Takahashi':
                final_next_H_T = final_hand_tuple # Takahashi's hand after playing and taking
                final_next_H_A = H_A            # Aoki's hand unchanged
            else: # player == 'Aoki'
                final_next_H_T = H_T            # Takahashi's hand unchanged
                final_next_H_A = final_hand_tuple # Aoki's hand after playing and taking
            
            # The next player remains the same as in Option 0 (the opponent).
            # The variable `next_player` is already set correctly above.
            
            # Recursive call for the opponent's turn from this resulting state.
            if not can_win(final_next_H_T, final_next_H_A, final_T_tuple, next_player):
                # Cache the result: current state is winning.
                memo[state_key] = True 
                return True # Found a winning move.

    # If the loop completes without returning True, it means none of the possible moves guarantee a win.
    # Therefore, the current player cannot force a win from this state (it's a losing state).
    memo[state_key] = False # Cache the result: current state is losing.
    return False

# Call the main function `solve()` to read input, run the game simulation, and print the output.
solve()