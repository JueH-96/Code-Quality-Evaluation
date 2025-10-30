# YOUR CODE HERE
import sys

def solve():
    """
    This function solves the Piano problem using dynamic programming.
    """
    # Read input from stdin
    try:
        # Use fast I/O in competitive programming environments
        input = sys.stdin.readline
        N_str = input()
        if not N_str: return
        N = int(N_str)
        notes = []
        for _ in range(N):
            a, s = input().strip().split()
            notes.append((int(a), s))
    except (IOError, ValueError):
        # Fallback for empty input or local testing
        return

    # K is the number of keys. We use 101 for 1-based indexing (keys 1-100)
    K = 101
    # Use a large number for infinity
    INF = float('inf')

    # dp_prev[l][r]: min fatigue with left hand at key l, right hand at key r
    # This table stores the states after the previous note press.
    dp_prev = [[INF] * K for _ in range(K)]

    # --- Base Case: First note ---
    A0, S0 = notes[0]
    if S0 == 'L':
        # Left hand must be at A0. Right hand can be placed at any key 'r' for free.
        for r in range(1, K):
            dp_prev[A0][r] = 0
    else:  # S0 == 'R'
        # Right hand must be at A0. Left hand can be placed at any key 'l' for free.
        for l in range(1, K):
            dp_prev[l][A0] = 0

    # --- DP Transitions: Subsequent notes ---
    # Iterate from the second note to the last.
    for i in range(1, N):
        dp_curr = [[INF] * K for _ in range(K)]
        Ai, Si = notes[i]

        if Si == 'L':
            # The hand to move is Left, and it must go to key Ai.
            # The Right hand's position 'r' remains unchanged from the previous step.
            for r in range(1, K):
                # We need to find min_{l_prev} (dp_prev[l_prev][r] + abs(Ai - l_prev)).
                min_cost_for_this_r = INF
                for l_prev in range(1, K):
                    if dp_prev[l_prev][r] != INF:
                        cost = dp_prev[l_prev][r] + abs(Ai - l_prev)
                        min_cost_for_this_r = min(min_cost_for_this_r, cost)
                
                if min_cost_for_this_r != INF:
                    dp_curr[Ai][r] = min_cost_for_this_r

        else:  # Si == 'R'
            # The hand to move is Right, and it must go to key Ai.
            # The Left hand's position 'l' remains unchanged.
            for l in range(1, K):
                # We need to find min_{r_prev} (dp_prev[l][r_prev] + abs(Ai - r_prev)).
                min_cost_for_this_l = INF
                for r_prev in range(1, K):
                    if dp_prev[l][r_prev] != INF:
                        cost = dp_prev[l][r_prev] + abs(Ai - r_prev)
                        min_cost_for_this_l = min(min_cost_for_this_l, cost)
                
                if min_cost_for_this_l != INF:
                    dp_curr[l][Ai] = min_cost_for_this_l
        
        # The current DP table becomes the previous one for the next iteration.
        dp_prev = dp_curr

    # --- Final Answer ---
    # The final answer is the minimum fatigue over all possible final hand positions.
    min_fatigue = INF
    for row in dp_prev:
        min_fatigue = min(min_fatigue, min(row))

    print(min_fatigue)

solve()