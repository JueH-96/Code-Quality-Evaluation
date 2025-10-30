# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    X = int(data[0])
    Y = int(data[1])
    Z = int(data[2])
    S = data[3]
    
    n = len(S)
    # Initialize DP table
    # dp[i][state] where state is 0 (Caps off) or 1 (Caps on)
    dp = [[float('inf')] * 2 for _ in range(n+1)]
    dp[0][0] = 0  # Start with Caps off
    
    for i in range(n):
        current_char = S[i]
        for state in range(2):
            if dp[i][state] == float('inf'):
                continue
            # Option 1: Press 'a' key only
            if (state == 0 and current_char == 'a') or (state == 1 and current_char == 'A'):
                dp[i+1][state] = min(dp[i+1][state], dp[i][state] + X)
            # Option 2: Press 'a' and Shift simultaneously
            if (state == 0 and current_char == 'A') or (state == 1 and current_char == 'a'):
                dp[i+1][state] = min(dp[i+1][state], dp[i][state] + Y)
            # Option 3: Toggle Caps Lock
            new_state = 1 - state
            if (new_state == 0 and current_char == 'a') or (new_state == 1 and current_char == 'A'):
                dp[i+1][new_state] = min(dp[i+1][new_state], dp[i][state] + Z + X)
            if (new_state == 0 and current_char == 'A') or (new_state == 1 and current_char == 'a'):
                dp[i+1][new_state] = min(dp[i+1][new_state], dp[i][state] + Z + Y)
    
    result = min(dp[n][0], dp[n][1])
    print(result)

if __name__ == "__main__":
    main()