# YOUR CODE HERE
n = int(input())
keys = []
for _ in range(n):
    line = input().split()
    keys.append((int(line[0]), line[1]))

# dp[i][l][r] = minimum fatigue after pressing first i keys
# with left hand at position l and right hand at position r
# We use -1 to indicate that a hand hasn't been placed yet
INF = float('inf')

# Initialize DP table
# Since keys are 1-100, we need indices 0-100 (101 total)
# We also use 0 to represent "not placed yet"
dp = {}

# Base case: no keys pressed yet, hands can be anywhere
dp[0] = {(0, 0): 0}  # Both hands not placed yet

for i in range(n):
    key, hand = keys[i]
    new_dp = {}
    
    for (left_pos, right_pos), fatigue in dp[i].items():
        if hand == 'L':
            # Need to press with left hand
            if left_pos == 0:  # Left hand not placed yet
                # Place left hand on current key
                new_state = (key, right_pos)
                if new_state not in new_dp:
                    new_dp[new_state] = INF
                new_dp[new_state] = min(new_dp[new_state], fatigue)
            else:
                # Move left hand from left_pos to key
                new_fatigue = fatigue + abs(key - left_pos)
                new_state = (key, right_pos)
                if new_state not in new_dp:
                    new_dp[new_state] = INF
                new_dp[new_state] = min(new_dp[new_state], new_fatigue)
        else:  # hand == 'R'
            # Need to press with right hand
            if right_pos == 0:  # Right hand not placed yet
                # Place right hand on current key
                new_state = (left_pos, key)
                if new_state not in new_dp:
                    new_dp[new_state] = INF
                new_dp[new_state] = min(new_dp[new_state], fatigue)
            else:
                # Move right hand from right_pos to key
                new_fatigue = fatigue + abs(key - right_pos)
                new_state = (left_pos, key)
                if new_state not in new_dp:
                    new_dp[new_state] = INF
                new_dp[new_state] = min(new_dp[new_state], new_fatigue)
    
    dp[i + 1] = new_dp

# Find minimum fatigue after all keys are pressed
min_fatigue = INF
for (left_pos, right_pos), fatigue in dp[n].items():
    min_fatigue = min(min_fatigue, fatigue)

print(min_fatigue)