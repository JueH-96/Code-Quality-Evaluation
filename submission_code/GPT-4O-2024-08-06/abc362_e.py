# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # dp[i][k] will store the number of arithmetic subsequences of length k ending at index i
    dp = [[0] * (N + 1) for _ in range(N)]
    
    # Each element itself is a subsequence of length 1
    for i in range(N):
        dp[i][1] = 1
    
    # Result array to store the number of arithmetic subsequences of each length
    result = [0] * (N + 1)
    
    # Count subsequences of length 1
    result[1] = N
    
    # Dictionary to store the count of differences
    from collections import defaultdict
    
    # Iterate over all pairs (i, j) with i < j
    for j in range(N):
        count = defaultdict(int)
        for i in range(j):
            d = A[j] - A[i]
            # For each k, update dp[j][k+1] using dp[i][k]
            for k in range(1, N):
                dp[j][k + 1] = (dp[j][k + 1] + dp[i][k]) % MOD
            # Update the count for this difference
            count[d] = (count[d] + dp[i][1]) % MOD
        
        # Update dp[j][2] using the count of differences
        for d in count:
            dp[j][2] = (dp[j][2] + count[d]) % MOD
    
    # Sum up the results for each k
    for k in range(2, N + 1):
        for i in range(N):
            result[k] = (result[k] + dp[i][k]) % MOD
    
    # Print the results for k = 1 to N
    print(" ".join(map(str, result[1:N+1])))