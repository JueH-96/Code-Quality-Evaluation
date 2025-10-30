# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    N = int(data[0])
    S = data[1]
    
    MOD = 998244353
    
    # Balance array to track the difference between W and B
    balance = [0] * (2 * N + 1)
    
    # Calculate balance
    for i in range(1, 2 * N + 1):
        balance[i] = balance[i - 1] + (1 if S[i - 1] == 'W' else -1)
    
    # Check if the balance is ever negative
    if min(balance) < 0:
        print(0)
        return
    
    # Dynamic programming to count valid pairings
    dp = [0] * (N + 1)
    dp[0] = 1  # Base case: one way to pair zero pairs
    
    # Count the number of ways to pair up to i pairs
    for i in range(1, 2 * N + 1):
        if S[i - 1] == 'W':
            for j in range(N, 0, -1):
                dp[j] = (dp[j] + dp[j - 1]) % MOD
        else:
            for j in range(N):
                dp[j] = dp[j + 1]
            dp[N] = 0
    
    # The result is the number of ways to pair all N pairs
    print(dp[N])