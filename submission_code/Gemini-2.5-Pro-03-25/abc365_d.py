# YOUR CODE HERE
import sys
import math

# Function to solve the problem
def solve():
    # Read N (number of games) from standard input
    N = int(sys.stdin.readline())
    # Read Aoki's sequence of moves S from standard input
    S = sys.stdin.readline().strip()

    # Define the possible moves in rock-paper-scissors
    moves = ['R', 'P', 'S']
    
    # Define a mapping from a move to the move that beats it
    # Rock (R) is beaten by Paper (P)
    # Paper (P) is beaten by Scissors (S)
    # Scissors (S) is beaten by Rock (R)
    win_map = {'R': 'P', 'P': 'S', 'S': 'R'}
    
    # Define a mapping from a move to the move that loses to it
    # Rock (R) beats Scissors (S)
    # Paper (P) beats Rock (R)
    # Scissors (S) beats Paper (P)
    lose_map = {'R': 'S', 'P': 'R', 'S': 'P'}

    # Initialize DP table for the previous step (conceptually, before game 1).
    # dp_prev[k] stores the maximum number of wins up to the previous game, 
    # ending with move `moves[k]`.
    # We use float('-inf') to represent states that are unreachable or invalid according to the problem conditions.
    # The indices correspond to moves: 0 for 'R', 1 for 'P', 2 for 'S'.
    dp_prev = [float('-inf')] * 3 

    # Base case: Calculate DP values for the first game (game index 1, string index 0)
    A1 = S[0] # Aoki's move in the first game
    # Iterate through Takahashi's possible moves ('R', 'P', 'S') for the first game
    for m_idx, m in enumerate(moves): 
        
        # Check Condition 1: Takahashi's move 'm' must not lose to Aoki's move 'A1'.
        # A move 'm' loses to 'A1' if 'm' is the move that 'A1' beats.
        # The move that 'A1' beats is `lose_map[A1]`.
        if m != lose_map[A1]: 
             # If the move 'm' is allowed (doesn't lose), check if it wins against 'A1'.
             is_win = (m == win_map[A1])
             # The score for the first game is 1 if Takahashi wins, and 0 if it's a draw.
             # Store this score in the DP table for the first step.
             dp_prev[m_idx] = 1 if is_win else 0

    # Dynamic Programming loop: Calculate DP values for games 2 to N.
    # The loop iterates from i = 1 to N-1. The game index is i+1, corresponding to string index i.
    for i in range(1, N):
        # Initialize DP table for the current step (game i+1).
        dp_curr = [float('-inf')] * 3
        Ai = S[i] # Aoki's move in the current game (game i+1)
        
        # Iterate through Takahashi's possible moves 'm' for the current game.
        for m_idx, m in enumerate(moves): 
            
            # Check Condition 1: Takahashi's current move 'm' must not lose to Aoki's move 'Ai'.
            if m != lose_map[Ai]: 
                
                # If move 'm' is allowed, calculate the maximum possible wins ending with this move.
                # This depends on the maximum wins achieved up to the previous game (game i),
                # ending with a move 'prev_m' that is different from the current move 'm' (Condition 2).
                max_prev_wins = float('-inf')
                # Iterate through possible moves 'prev_m' from the previous game.
                for prev_m_idx, prev_m in enumerate(moves): 
                    
                    # Check Condition 2: Takahashi's current move 'm' must be different from the previous move 'prev_m'.
                    if m != prev_m: 
                        
                        # Check if the state ending with 'prev_m' in the previous game was reachable (score is not -inf).
                        if dp_prev[prev_m_idx] != float('-inf'): 
                            # Update the maximum score attainable from a valid previous state that satisfies Condition 2.
                            max_prev_wins = max(max_prev_wins, dp_prev[prev_m_idx])

                # If there was at least one valid path (reachable previous state satisfying conditions) leading to the current state (m_idx).
                # The problem guarantees that such a path always exists for at least one move `m`, so `max_prev_wins` will eventually be non-infinite for some `m_idx`.
                if max_prev_wins != float('-inf'): 
                    # Check if the current move 'm' results in a win against 'Ai'.
                    is_win = (m == win_map[Ai])
                    # Calculate the total wins for the current state: max wins from a valid previous state + 1 (if current move wins) or 0 (if draw).
                    current_wins = max_prev_wins + (1 if is_win else 0)
                    # Store the calculated maximum wins for the state ending with move 'm' at game i+1.
                    dp_curr[m_idx] = current_wins
        
        # Update the 'previous step' DP table for the next iteration.
        # The DP values for the current game become the DP values for the 'previous' game in the next loop cycle.
        dp_prev = dp_curr

    # After processing all N games, the final answer is the maximum value among all reachable states in the last DP table 'dp_prev'.
    final_max_wins = 0 # The minimum possible number of wins is 0. Initialize the overall maximum wins found.
    for wins in dp_prev:
       # Consider only reachable states (score is not -inf).
       if wins != float('-inf'): 
            # Update the overall maximum wins found across all possible ending moves for game N.
            final_max_wins = max(final_max_wins, wins)
    
    # Print the final maximum number of wins. Convert to integer for standard output format.
    print(int(final_max_wins))

# Execute the solve function to run the program
solve()