def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    X = int(data[0])
    Y = int(data[1])
    Z = int(data[2])
    S = data[3]
    
    N = len(S)
    dp = [[0, 0] for _ in range(N + 1)]
    
    # Base cases
    dp[0][0] = 0  # Starting with Caps Lock off
    dp[0][1] = Z  # Starting with Caps Lock on (after toggling)
    
    for i in range(1, N + 1):
        char = S[i - 1]
        # Option 1: Stay in current state and type the character
        # State 0
        cost_if_stay_0 = dp[i - 1][0]
        if char == 'a':
            cost_if_stay_0 += X
        else:  # char == 'A'
            cost_if_stay_0 += Y
        dp[i][0] = cost_if_stay_0
        
        # State 1
        cost_if_stay_1 = dp[i - 1][1]
        if char == 'A':
            cost_if_stay_1 += X
        else:  # char == 'a'
            cost_if_stay_1 += Y
        dp[i][1] = cost_if_stay_1
        
        # Option 2: Toggle Caps Lock and type the character in the new state
        # Toggle from state 0 to 1
        cost_if_toggle_to_1 = dp[i - 1][0] + Z
        if char == 'A':
            cost_if_toggle_to_1 += X
        else:
            cost_if_toggle_to_1 += Y
        dp[i][1] = min(dp[i][1], cost_if_toggle_to_1)
        
        # Toggle from state 1 to 0
        cost_if_toggle_to_0 = dp[i - 1][1] + Z
        if char == 'a':
            cost_if_toggle_to_0 += X
        else:
            cost_if_toggle_to_0 += Y
        dp[i][0] = min(dp[i][0], cost_if_toggle_to_0)
    
    print(min(dp[N][0], dp[N][1]))

if __name__ == '__main__':
    main()