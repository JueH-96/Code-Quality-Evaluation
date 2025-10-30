X, Y, Z = map(int, input().split())
S = input().strip()

def cost_to_write_char(char, state):
    if state == 0:  # caps lock off
        return X if char == 'a' else Y
    else:  # state == 1, caps lock on
        return X if char == 'A' else Y

# dp[state] = min cost to produce current prefix and end up in state
dp = [0, Z]  # Initially: state 0 costs 0, state 1 costs Z (one toggle)

for char in S:
    new_dp = [float('inf'), float('inf')]
    
    # To end up in state 0
    # Option 1: Was in state 0, write char, stay in state 0
    new_dp[0] = min(new_dp[0], dp[0] + cost_to_write_char(char, 0))
    # Option 2: Was in state 1, write char, toggle to state 0
    new_dp[0] = min(new_dp[0], dp[1] + cost_to_write_char(char, 1) + Z)
    
    # To end up in state 1
    # Option 1: Was in state 0, toggle to state 1, write char
    new_dp[1] = min(new_dp[1], dp[0] + Z + cost_to_write_char(char, 1))
    # Option 2: Was in state 1, write char, stay in state 1
    new_dp[1] = min(new_dp[1], dp[1] + cost_to_write_char(char, 1))
    
    dp = new_dp

print(min(dp[0], dp[1]))