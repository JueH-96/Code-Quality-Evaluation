import sys

MAX_KEYS = 100
INF = float('inf')

# Read N
N = int(sys.stdin.readline())

# Read notes
notes_data = []
for _ in range(N):
    line = sys.stdin.readline().split()
    A_i = int(line[0])
    S_i = line[1]
    notes_data.append((A_i, S_i))

# dp[l_pos][r_pos] = min fatigue with left hand at l_pos, right hand at r_pos
# Initialize with 0, representing free initial placement.
# dp[0] row/column will be unused as keys are 1-indexed (1 to MAX_KEYS).
dp = [[0] * (MAX_KEYS + 1) for _ in range(MAX_KEYS + 1)]

for i in range(N):
    target_A, hand_S = notes_data[i]
    
    new_dp = [[INF] * (MAX_KEYS + 1) for _ in range(MAX_KEYS + 1)]
    
    if hand_S == 'L':
        # Left hand moves to target_A. Right hand stays at r_key.
        for r_key in range(1, MAX_KEYS + 1): # Current position of right hand (unchanged)
            # new_dp[target_A][r_key] will be the minimum fatigue to reach this state.
            # It's computed by considering all possible previous positions of the left hand.
            for prev_l_key in range(1, MAX_KEYS + 1): # Previous position of left hand
                if dp[prev_l_key][r_key] == INF: # If this previous state was unreachable
                    continue
                
                cost_to_move_L = abs(target_A - prev_l_key)
                current_total_fatigue = dp[prev_l_key][r_key] + cost_to_move_L
                
                # Update the cost for the new state (left hand at target_A, right hand at r_key) 
                # if this path is better
                if current_total_fatigue < new_dp[target_A][r_key]:
                    new_dp[target_A][r_key] = current_total_fatigue
    else: # hand_S == 'R'
        # Right hand moves to target_A. Left hand stays at l_key.
        for l_key in range(1, MAX_KEYS + 1): # Current position of left hand (unchanged)
            # new_dp[l_key][target_A] will be the minimum fatigue to reach this state.
            # It's computed by considering all possible previous positions of the right hand.
            for prev_r_key in range(1, MAX_KEYS + 1): # Previous position of right hand
                if dp[l_key][prev_r_key] == INF: # If this previous state was unreachable
                    continue
                
                cost_to_move_R = abs(target_A - prev_r_key)
                current_total_fatigue = dp[l_key][prev_r_key] + cost_to_move_R
                
                # Update the cost for the new state (left hand at l_key, right hand at target_A)
                # if this path is better
                if current_total_fatigue < new_dp[l_key][target_A]:
                    new_dp[l_key][target_A] = current_total_fatigue
    
    # The new_dp table for this step becomes the dp table for the next step
    dp = new_dp 

# After processing all notes, find the minimum fatigue in the final dp table.
# This corresponds to the minimum fatigue regardless of where hands end up.
min_total_fatigue_at_end = INF
for l_key_final in range(1, MAX_KEYS + 1):
    for r_key_final in range(1, MAX_KEYS + 1):
        if dp[l_key_final][r_key_final] < min_total_fatigue_at_end:
            min_total_fatigue_at_end = dp[l_key_final][r_key_final]

# Print the result
sys.stdout.write(str(min_total_fatigue_at_end) + "
")