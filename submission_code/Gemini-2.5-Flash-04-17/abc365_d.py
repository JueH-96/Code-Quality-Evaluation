import sys

# Helper functions for win and draw moves in rock-paper-scissors
def win(aoki_move):
    """Returns the move that beats aoki_move."""
    if aoki_move == 'R': return 'P' # Paper beats Rock
    if aoki_move == 'P': return 'S' # Scissors beats Paper
    if aoki_move == 'S': return 'R' # Rock beats Scissors

def draw(aoki_move):
    """Returns the move that draws against aoki_move."""
    return aoki_move # Same move results in a draw

def solve():
    # Read input
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    # Possible moves for Takahashi
    all_moves = ['R', 'P', 'S']

    # DP state: dp_prev[move] stores the maximum wins up to the previous game (i-1),
    # ending with Takahashi playing 'move' in game (i-1).
    # We use O(1) space by only storing the DP values for the previous game.
    dp_prev = {move: float('-inf') for move in all_moves}

    # Base case: Game 0 (index 0)
    aoki_0 = S[0]

    # Takahashi wins game 0 if playing win(aoki_0). Score = 1.
    dp_prev[win(aoki_0)] = 1

    # Takahashi draws game 0 if playing draw(aoki_0). Score = 0.
    # win(aoki_0) and draw(aoki_0) are always different moves for any aoki_move.
    # We use max here to ensure it's set to 0, as initially it's -inf.
    dp_prev[draw(aoki_0)] = max(dp_prev[draw(aoki_0)], 0)

    # Iterate through games from 1 to N-1
    for i in range(1, N):
        aoki_i = S[i]
        # Valid moves for Takahashi in game i are those that don't lose
        valid_moves_i = {win(aoki_i), draw(aoki_i)}

        # dp_curr will store the DP values for the current game i
        dp_curr = {move: float('-inf') for move in all_moves}

        # Calculate maximum DP values from the previous step (i-1),
        # excluding a specific move. This helps calculate transitions quickly.
        # max(dp_prev['P'], dp_prev['S']) = max wins from previous game ending in P or S
        # max(dp_prev['R'], dp_prev['S']) = max wins from previous game ending in R or S
        # max(dp_prev['R'], dp_prev['P']) = max wins from previous game ending in R or P
        max_dp_prev_not_R = max(dp_prev['P'], dp_prev['S'])
        max_dp_prev_not_P = max(dp_prev['R'], dp_prev['S'])
        max_dp_prev_not_S = max(dp_prev['R'], dp_prev['P'])

        # Consider each possible move ('R', 'P', 'S') for Takahashi in the current game
        for curr_move in all_moves:
            # Condition 1: Takahashi must play a move that does not lose to Aoki
            if curr_move in valid_moves_i:
                # Determine the score increase for the current game (1 for win, 0 for draw)
                score_increase = 1 if curr_move == win(aoki_i) else 0

                # Condition 2: Takahashi's move must be different from the previous game's move.
                # Find the maximum wins from the previous game (dp_prev) where the previous move
                # (prev_move) is different from the current move (curr_move).
                max_prev_dp = float('-inf')
                if curr_move == 'R':
                    max_prev_dp = max_dp_prev_not_R
                elif curr_move == 'P':
                    max_prev_dp = max_dp_prev_not_P
                elif curr_move == 'S':
                    max_prev_dp = max_dp_prev_not_S

                # Based on the problem guarantee, max_prev_dp will always be >= 0 if curr_move is valid
                # in game i > 0, as there must be a valid previous state it can transition from.
                # So we can directly add the score increase.
                dp_curr[curr_move] = max_prev_dp + score_increase

        # The DP values for the current game become the previous values for the next iteration
        dp_prev = dp_curr

    # After iterating through all games (up to N-1), the maximum value in the final
    # DP state (dp_prev) represents the maximum number of games Takahashi could have won.
    # The problem guarantees a solution exists, so max(dp_prev.values()) will be >= 0.
    max_wins = max(dp_prev.values())

    # Print the result
    print(max_wins)

# Execute the solve function
solve()