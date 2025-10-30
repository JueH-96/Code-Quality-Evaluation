MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    
    P = []
    V = []
    
    index = 2
    for _ in range(Q):
        P.append(int(data[index]))
        V.append(int(data[index + 1]))
        index += 2
    
    # We will use dynamic programming to solve this problem.
    # dp[i][j] will represent the number of ways to perform operations up to the i-th operation
    # such that the last operation type was j (0 for prefix, 1 for suffix).
    dp = [[0, 0] for _ in range(Q)]
    
    # Initialize the first operation possibilities
    dp[0][0] = 1  # Can always set the prefix to V[0]
    dp[0][1] = 1  # Can always set the suffix to V[0]
    
    # We need to keep track of the minimum and maximum values possible for each position in S
    # after each operation to determine if Snuke will cry or not.
    min_prefix = [float('inf')] * (N + 1)
    max_prefix = [-float('inf')] * (N + 1)
    min_suffix = [float('inf')] * (N + 1)
    max_suffix = [-float('inf')] * (N + 1)
    
    # Initial conditions after the first operation
    for i in range(1, P[0] + 1):
        min_prefix[i] = V[0]
        max_prefix[i] = V[0]
    for i in range(P[0], N + 1):
        min_suffix[i] = V[0]
        max_suffix[i] = V[0]
    
    # Process each operation from the second to the last
    for i in range(1, Q):
        p = P[i]
        v = V[i]
        
        # Option 1: Set prefix [1, p] to v
        # Check if it's valid to come from the previous prefix setting
        if max_prefix[p] <= v:
            dp[i][0] = (dp[i][0] + dp[i-1][0]) % MOD
        # Check if it's valid to come from the previous suffix setting
        if max_suffix[p] <= v:
            dp[i][0] = (dp[i][0] + dp[i-1][1]) % MOD
        
        # Option 2: Set suffix [p, N] to v
        # Check if it's valid to come from the previous prefix setting
        if min_prefix[p] <= v:
            dp[i][1] = (dp[i][1] + dp[i-1][0]) % MOD
        # Check if it's valid to come from the previous suffix setting
        if min_suffix[p] <= v:
            dp[i][1] = (dp[i][1] + dp[i-1][1]) % MOD
        
        # Update min/max values for prefix and suffix
        for j in range(1, p + 1):
            min_prefix[j] = min(min_prefix[j], v)
            max_prefix[j] = max(max_prefix[j], v)
        for j in range(p, N + 1):
            min_suffix[j] = min(min_suffix[j], v)
            max_suffix[j] = max(max_suffix[j], v)
    
    # The answer is the sum of ways to perform all operations without Snuke crying
    result = (dp[Q-1][0] + dp[Q-1][1]) % MOD
    print(result)

if __name__ == "__main__":
    main()