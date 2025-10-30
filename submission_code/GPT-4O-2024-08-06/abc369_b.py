# YOUR CODE HERE
def min_fatigue(n, actions):
    # Initialize DP table with infinity
    dp = [[[float('inf')] * 101 for _ in range(101)] for _ in range(n + 1)]
    
    # Initial state: no fatigue, hands can start on any key
    for l in range(1, 101):
        for r in range(1, 101):
            dp[0][l][r] = 0
    
    # Process each action
    for i in range(n):
        key, hand = actions[i]
        for l in range(1, 101):
            for r in range(1, 101):
                if hand == 'L':
                    # Move left hand to key
                    dp[i + 1][key][r] = min(dp[i + 1][key][r], dp[i][l][r] + abs(l - key))
                else:
                    # Move right hand to key
                    dp[i + 1][l][key] = min(dp[i + 1][l][key], dp[i][l][r] + abs(r - key))
    
    # Find the minimum fatigue after all actions
    min_fatigue = float('inf')
    for l in range(1, 101):
        for r in range(1, 101):
            min_fatigue = min(min_fatigue, dp[n][l][r])
    
    return min_fatigue

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().strip().split('
')
    
    n = int(data[0].strip())
    actions = []
    for i in range(1, n + 1):
        a, s = data[i].strip().split()
        actions.append((int(a), s))
    
    result = min_fatigue(n, actions)
    print(result)