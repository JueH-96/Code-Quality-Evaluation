import sys

def solve():
    """
    Solves the Rock-Paper-Scissors problem using dynamic programming.
    """
    # It's good practice to set a higher recursion limit for problems that might
    # involve deep recursion, though this iterative solution does not need it.
    # sys.setrecursionlimit(10**6)

    try:
        N = int(sys.stdin.readline())
        S = sys.stdin.readline().strip()
    except (ValueError, IndexError):
        # Handle cases with empty input on some platforms
        return

    # dp[move]: max wins up to the *previous* game, ending with 'move'.
    # move: 0 for R, 1 for P, 2 for S.
    # Initialize with -1 to represent unreachable states.
    dp = [-1, -1, -1]

    move_map = {'R': 0, 'P': 1, 'S': 2}

    def is_win(t_move, a_move):
        """Checks if Takahashi's move is a win. Returns 1 if win, 0 otherwise."""
        # A move `t` beats `a` if (t - a) mod 3 is 1 (with our R-P-S mapping).
        return 1 if (t_move - a_move + 3) % 3 == 1 else 0

    def is_allowed(t_move, a_move):
        """Checks if Takahashi's move is allowed (win or draw)."""
        # A move `t` loses to `a` if (t - a) mod 3 is 2.
        # Any other outcome is a win or a draw, which is allowed.
        return (t_move - a_move + 3) % 3 != 2

    # --- Base case: Game 0 (the first game) ---
    aoki_move_0 = move_map[S[0]]
    for t_move in range(3):
        if is_allowed(t_move, aoki_move_0):
            dp[t_move] = is_win(t_move, aoki_move_0)

    # --- DP transition for games 1 to N-1 ---
    for i in range(1, N):
        aoki_move = move_map[S[i]]
        
        # Store scores from the previous step before calculating new ones
        prev_dp = dp
        
        # Reset dp array to hold scores for the current game
        dp = [-1, -1, -1]  

        # Pre-calculate max wins from the previous step, ensuring the move is different
        max_prev_scores = [
            max(prev_dp[1], prev_dp[2]),  # To play R (0), prev must be P (1) or S (2)
            max(prev_dp[0], prev_dp[2]),  # To play P (1), prev must be R (0) or S (2)
            max(prev_dp[0], prev_dp[1])   # To play S (2), prev must be R (0) or P (1)
        ]

        # Iterate through Takahashi's possible moves for the current game
        for t_move in range(3):
            # A path is valid if the move is allowed and there was a valid preceding sequence
            if is_allowed(t_move, aoki_move) and max_prev_scores[t_move] != -1:
                win_points = is_win(t_move, aoki_move)
                dp[t_move] = win_points + max_prev_scores[t_move]
    
    # The final answer is the maximum score after N games, regardless of the last move.
    print(max(dp))

solve()