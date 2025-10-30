import sys

# Main function to encapsulate the solution logic
def solve():
    # Read N, the number of presses
    N = int(sys.stdin.readline())
    
    # Read the sequence of actions. Each action is a tuple (key_number, hand_used).
    actions = []
    for _ in range(N):
        A, S = sys.stdin.readline().split()
        actions.append((int(A), S))
    
    MAX_KEY = 100 # The maximum key number, as specified in the problem (1 to 100)
    
    # Initialize the DP table.
    # dp[l_pos][r_pos] will store the minimum fatigue accumulated after processing
    # a certain number of actions, with the left hand currently at key l_pos
    # and the right hand currently at key r_pos.
    # We use 1-based indexing for keys, so the array size is (MAX_KEY + 1) x (MAX_KEY + 1).
    # Initialize all states with float('inf') to represent unreachable states or
    # very high fatigue levels that will be minimized.
    dp = [[float('inf')] * (MAX_KEY + 1) for _ in range(MAX_KEY + 1)]
    
    # --- Base Case: Process the first action ---
    # The problem states that hands can be placed on any keys initially with 0 fatigue.
    # This implies that the cost for the very first press is 0, because we can
    # strategically place the required hand directly on the key to be pressed first.
    
    first_A, first_S = actions[0]
    if first_S == 'L':
        # If the first press is with the left hand, it must be placed on key `first_A`.
        # The right hand can be placed on any key from 1 to MAX_KEY.
        # The initial fatigue for these states is 0.
        for r_pos in range(1, MAX_KEY + 1):
            dp[first_A][r_pos] = 0
    else: # first_S == 'R'
        # If the first press is with the right hand, it must be placed on key `first_A`.
        # The left hand can be placed on any key from 1 to MAX_KEY.
        # The initial fatigue for these states is 0.
        for l_pos in range(1, MAX_KEY + 1):
            dp[l_pos][first_A] = 0
            
    # --- Iterative Step: Process subsequent actions (from the second action onwards) ---
    # We iterate from the second action (index 1) up to the last action (index N-1).
    for i in range(1, N):
        current_A, current_S = actions[i]
        
        # Create a new DP table for the results of the current iteration.
        # This is crucial to ensure that calculations for `next_dp` are based
        # entirely on the states from `dp` (i.e., from the previous action),
        # preventing incorrect dependencies on partially updated states within the same iteration.
        next_dp = [[float('inf')] * (MAX_KEY + 1) for _ in range(MAX_KEY + 1)]
        
        # Iterate over all possible previous hand positions (prev_L for left, prev_R for right).
        for prev_L in range(1, MAX_KEY + 1):
            for prev_R in range(1, MAX_KEY + 1):
                # If the current state `dp[prev_L][prev_R]` was unreachable (fatigue is infinity),
                # there's no way to reach new states from it, so we skip it.
                if dp[prev_L][prev_R] == float('inf'):
                    continue
                
                # Calculate the new fatigue level based on the current action (current_A, current_S).
                if current_S == 'L':
                    # The left hand needs to move from its previous position (prev_L) to `current_A`.
                    # The right hand stays at its previous position (prev_R).
                    fatigue_increase = abs(current_A - prev_L)
                    new_fatigue = dp[prev_L][prev_R] + fatigue_increase
                    
                    # Update the `next_dp` table for the new state (left hand at current_A, right hand at prev_R).
                    # We take the minimum, as multiple previous states might lead to this new state.
                    next_dp[current_A][prev_R] = min(next_dp[current_A][prev_R], new_fatigue)
                else: # current_S == 'R'
                    # The right hand needs to move from its previous position (prev_R) to `current_A`.
                    # The left hand stays at its previous position (prev_L).
                    fatigue_increase = abs(current_A - prev_R)
                    new_fatigue = dp[prev_L][prev_R] + fatigue_increase
                    
                    # Update the `next_dp` table for the new state (left hand at prev_L, right hand at current_A).
                    next_dp[prev_L][current_A] = min(next_dp[prev_L][current_A], new_fatigue)
        
        # After calculating all possible states for the current action `i`,
        # update `dp` to `next_dp` for the next iteration (action `i+1`).
        dp = next_dp
        
    # --- Final Result ---
    # After processing all N actions, the minimum total fatigue is the smallest value
    # found anywhere in the final `dp` table, across all possible ending positions
    # for the left and right hands.
    min_total_fatigue = float('inf')
    for l_pos in range(1, MAX_KEY + 1):
        for r_pos in range(1, MAX_KEY + 1):
            min_total_fatigue = min(min_total_fatigue, dp[l_pos][r_pos])
            
    # Print the minimum total fatigue. Since fatigue levels are integers, cast to int.
    print(int(min_total_fatigue))

# Call the solve function to execute the program
solve()