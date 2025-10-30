# YOUR CODE HERE
import sys

def solve():
    N = int(sys.stdin.readline())
    notes = []
    for _ in range(N):
        A, S = sys.stdin.readline().split()
        notes.append((int(A), S))

    # DP table dp[k][l][r] = min fatigue after k notes, left hand at l, right hand at r
    # Using 2 tables for space optimization: dp[0][l][r] and dp[1][l][r]
    # Keys are 1 to 100. Use 1-based indexing for convenience. Size 101x101.
    MAX_KEY = 100
    INF = float('inf') # Use a large float for infinity

    # Initialize dp tables
    # dp[k % 2][l][r] stores min fatigue after processing k notes, L at l, R at r
    # Initialize with INF for all states
    dp = [[([INF] * (MAX_KEY + 1)) for _ in range(MAX_KEY + 1)] for _ in range(2)]

    # Base case: Before any notes (k=0), fatigue is 0 for any hand positions
    # dp[0][l][r] = 0 for all 1 <= l, r <= 100
    for l in range(1, MAX_KEY + 1):
        for r in range(1, MAX_KEY + 1):
            dp[0][l][r] = 0

    # Iterate through notes from k=0 to N-1
    # current_idx stores the DP table for step k (state after k notes)
    # next_idx stores the DP table for step k+1 (state after k+1 notes)
    # Use k % 2 to alternate between the two tables
    for k in range(N):
        A, S = notes[k]
        target = A
        current_idx = k % 2 # Index of the DP table after k notes
        next_idx = (k + 1) % 2 # Index of the DP table after k+1 notes

        # Initialize the DP table for step k+1 with infinity
        for l in range(1, MAX_KEY + 1):
            for r in range(1, MAX_KEY + 1):
                dp[next_idx][l][r] = INF

        if S == 'L':
            # Left hand presses target key A. Right hand stays at its previous position.
            # A state (k+1, target, r_after) is reachable from (k, l_before, r_before)
            # where r_after = r_before.
            # The cost added is |target - l_before|.
            # dp[k+1][target][r_before] = min_{l_before} (dp[k][l_before][r_before] + |target - l_before|)

            # Calculate min_cost_L_move[r_before] = min_{l_before} (dp[current_idx][l_before][r_before] + |target - l_before|)
            # This array stores the minimum fatigue to reach a state after step k, where the left hand moved to 'target',
            # and the right hand is at r_before, minimized over all possible previous left hand positions l_before.
            min_cost_L_move = [INF] * (MAX_KEY + 1) # Use 1-based indexing
            for r_before in range(1, MAX_KEY + 1):
                for l_before in range(1, MAX_KEY + 1):
                    if dp[current_idx][l_before][r_before] != INF:
                         min_cost_L_move[r_before] = min(min_cost_L_move[r_before], dp[current_idx][l_before][r_before] + abs(target - l_before))

            # Update the next DP table: dp[next_idx][target][r_after] = min_cost_L_move[r_before]
            # Since the right hand stayed put, r_after = r_before.
            for r_after in range(1, MAX_KEY + 1):
                 # If min_cost_L_move[r_after] is not INF, it means it's possible to reach this state
                 if min_cost_L_move[r_after] != INF:
                     dp[next_idx][target][r_after] = min_cost_L_move[r_after]

        elif S == 'R':
            # Right hand presses target key A. Left hand stays at its previous position.
            # A state (k+1, l_after, target) is reachable from (k, l_before, r_before)
            # where l_after = l_before.
            # The cost added is |target - r_before|.
            # dp[k+1][l_before][target] = min_{r_before} (dp[k][l_before][r_before] + |target - r_before|)

            # Calculate min_cost_R_move[l_before] = min_{r_before} (dp[current_idx][l_before][r_before] + |target - r_before|)
            # This array stores the minimum fatigue to reach a state after step k, where the right hand moved to 'target',
            # and the left hand is at l_before, minimized over all possible previous right hand positions r_before.
            min_cost_R_move = [INF] * (MAX_KEY + 1) # Use 1-based indexing
            for l_before in range(1, MAX_KEY + 1):
                for r_before in range(1, MAX_KEY + 1):
                     if dp[current_idx][l_before][r_before] != INF:
                         min_cost_R_move[l_before] = min(min_cost_R_move[l_before], dp[current_idx][l_before][r_before] + abs(target - r_before))

            # Update the next DP table: dp[next_idx][l_after][target] = min_cost_R_move[l_before]
            # Since the left hand stayed put, l_after = l_before.
            for l_after in range(1, MAX_KEY + 1):
                # If min_cost_R_move[l_after] is not INF, it means it's possible to reach this state
                if min_cost_R_move[l_after] != INF:
                    dp[next_idx][l_after][target] = min_cost_R_move[l_after]

        # The DP table for step k+1 is now computed in dp[next_idx].
        # For the next iteration (k+1), this table will become the 'current' table.
        # This is handled by the k%2 and (k+1)%2 logic in the loop.

    # The minimum fatigue is the minimum value across all possible hand positions
    # after the last note (N-th note) has been played.
    final_dp_idx = N % 2 # Index of the DP table after N notes
    min_fatigue = INF
    for l in range(1, MAX_KEY + 1):
        for r in range(1, MAX_KEY + 1):
            min_fatigue = min(min_fatigue, dp[final_dp_idx][l][r])

    print(min_fatigue)

solve()