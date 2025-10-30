n = int(input())
presses = []
for _ in range(n):
    line = input().split()
    key = int(line[0])
    hand = line[1]
    presses.append((key, hand))

# dp[left][right] = minimum fatigue with left hand at left, right hand at right
# We'll use two dp arrays to represent current and next state
prev_dp = {}
prev_dp[(0, 0)] = 0  # dummy initial state, we'll handle first press specially

for i, (key, hand) in enumerate(presses):
    curr_dp = {}
    
    if i == 0:
        # First press - we can place hands anywhere initially
        if hand == 'L':
            # Left hand presses key, right hand can be anywhere
            for right_pos in range(1, 101):
                curr_dp[(key, right_pos)] = 0
        else:
            # Right hand presses key, left hand can be anywhere  
            for left_pos in range(1, 101):
                curr_dp[(left_pos, key)] = 0
    else:
        # Subsequent presses
        for (prev_left, prev_right), prev_cost in prev_dp.items():
            if hand == 'L':
                # Left hand moves to key, right hand stays
                new_cost = prev_cost + abs(key - prev_left)
                state = (key, prev_right)
                if state not in curr_dp or curr_dp[state] > new_cost:
                    curr_dp[state] = new_cost
            else:
                # Right hand moves to key, left hand stays
                new_cost = prev_cost + abs(key - prev_right)
                state = (prev_left, key)
                if state not in curr_dp or curr_dp[state] > new_cost:
                    curr_dp[state] = new_cost
    
    prev_dp = curr_dp

# Find minimum fatigue among all final states
result = min(prev_dp.values())
print(result)