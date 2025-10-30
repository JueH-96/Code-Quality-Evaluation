import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    # WIN_AGAINST_AOKI[aoki_move] gives the move Takahashi needs to play to WIN against aoki_move.
    WIN_AGAINST_AOKI = {
        'R': 'P', # Takahashi plays Paper to beat Aoki's Rock
        'P': 'S', # Takahashi plays Scissors to beat Aoki's Paper
        'S': 'R'  # Takahashi plays Rock to beat Aoki's Scissors
    }
    ALL_MOVES = ['R', 'P', 'S']

    # dp[move_char] stores the maximum wins up to the current game,
    # with Takahashi playing 'move_char' in the current game.
    # Initialize with negative infinity to represent impossible states.
    prev_dp = {'R': -float('inf'), 'P': -float('inf'), 'S': -float('inf')}

    # Base case: i = 0 (first game)
    aoki_move_0 = S[0]
    for t_move in ALL_MOVES:
        if t_move == aoki_move_0:  # Takahashi draws
            prev_dp[t_move] = 0
        elif WIN_AGAINST_AOKI[aoki_move_0] == t_move:  # Takahashi wins
            prev_dp[t_move] = 1
        # Else (Takahashi loses), it remains -float('inf'), meaning this state is unreachable

    # Iterate from i = 1 to N-1 (subsequent games)
    for i in range(1, N):
        aoki_move_i = S[i]
        # Initialize current_dp for the current game
        current_dp = {'R': -float('inf'), 'P': -float('inf'), 'S': -float('inf')}

        for current_t_move in ALL_MOVES:
            current_score = 0
            if current_t_move == aoki_move_i:  # Takahashi draws
                current_score = 0
            elif WIN_AGAINSI_AOKI[aoki_move_i] == current_t_move:  # Takahashi wins
                current_score = 1
            else:  # Takahashi loses, this move is impossible for the current game
                continue # current_dp[current_t_move] remains -float('inf')

            # Find max wins from previous game, ensuring T_i-1 != T_i
            max_prev_wins = -float('inf')
            for prev_t_move in ALL_MOVES:
                if prev_t_move != current_t_move:
                    max_prev_wins = max(max_prev_wins, prev_dp[prev_t_move])
            
            # If a valid path existed from the previous game, update current_dp
            if max_prev_wins != -float('inf'):
                current_dp[current_t_move] = current_score + max_prev_wins
        
        # After computing current_dp for all moves in game i,
        # it becomes the prev_dp for the next iteration.
        prev_dp = current_dp.copy() 

    # The problem guarantees that a valid sequence of moves exists.
    # Therefore, the maximum value in prev_dp will not be -float('inf').
    max_wins = 0
    # Use max() directly on values for cleaner code
    max_wins = max(prev_dp.values())
    
    sys.stdout.write(str(max_wins) + "
")

solve()