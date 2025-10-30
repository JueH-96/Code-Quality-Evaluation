import sys

def solve():
    N = int(sys.stdin.readline())
    S_str = sys.stdin.readline().strip()

    # Map R, P, S to 0, 1, 2
    # R=0, P=1, S=2
    # T beats A if T == (A+1)%3. (e.g. P(1) beats R(0) since 1 == (0+1)%3)
    # T draws A if T == A.
    # T loses to A if T == (A-1+3)%3. (e.g. S(2) loses to R(0) since 2 == (0-1+3)%3)
    
    char_to_int = {'R': 0, 'P': 1, 'S': 2}
    aoki_moves_int = [char_to_int[c] for c in S_str]

    # dp[j] stores max wins up to the current game, with Takahashi's last move being j.
    # Initialize with -1 to signify an invalid/impossible path.
    dp = [-1, -1, -1]

    # Base case: Game 0 (index 0)
    aoki_0 = aoki_moves_int[0]
    
    for takahashi_move_0 in range(3): # Iterate R, P, S for Takahashi
        # Check if takahashi_move_0 loses to aoki_0
        if takahashi_move_0 == (aoki_0 - 1 + 3) % 3:
            # dp[takahashi_move_0] remains -1 (invalid state)
            pass
        elif takahashi_move_0 == (aoki_0 + 1) % 3: # Takahashi wins
            dp[takahashi_move_0] = 1
        else: # Takahashi draws (takahashi_move_0 == aoki_0)
            dp[takahashi_move_0] = 0
            
    # Iterate for games 1 to N-1 (indices 1 to N-1)
    for i in range(1, N):
        aoki_i = aoki_moves_int[i]
        new_dp = [-1, -1, -1] # Stores DP values for game i
        
        for curr_t_move in range(3): # Takahashi's move for current game i
            # Condition 1: Takahashi never lost.
            # If curr_t_move loses to aoki_i, it's not a valid move.
            if curr_t_move == (aoki_i - 1 + 3) % 3:
                # new_dp[curr_t_move] remains -1
                continue

            # Calculate score for this turn (1 if win, 0 if draw)
            score_this_turn = 0
            if curr_t_move == (aoki_i + 1) % 3: # curr_t_move wins
                score_this_turn = 1
            
            # Condition 2: Takahashi's move in game i-1 is different from game i.
            max_prev_dp_val = -1 # Max wins from game i-1 ending in a different move
            
            for prev_t_move in range(3): # Takahashi's move for game i-1
                if prev_t_move == curr_t_move: # Move must be different
                    continue
                
                if dp[prev_t_move] != -1: # If path ending in prev_t_move was possible
                    if max_prev_dp_val == -1 or dp[prev_t_move] > max_prev_dp_val:
                         max_prev_dp_val = dp[prev_t_move]
            
            if max_prev_dp_val != -1: # If there's a valid path from previous game
                new_dp[curr_t_move] = max_prev_dp_val + score_this_turn
            # Else, new_dp[curr_t_move] remains -1 (unreachable from valid previous states)

        dp = new_dp # Update dp table for the next iteration
        
    # The final answer is the maximum number of wins ending in any move in the last game.
    # Problem guarantees a solution exists, so max(dp) will be >= 0.
    ans = 0
    if N > 0 : # N >= 1 as per constraints
        ans = max(dp)
        # Defensive check, though problem guarantees max(dp) >= 0.
        if ans == -1:
             ans = 0


    sys.stdout.write(str(ans) + "
")

solve()