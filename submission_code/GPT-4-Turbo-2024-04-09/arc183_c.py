MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    constraints = []
    index = 2
    for _ in range(M):
        L = int(data[index]) - 1
        R = int(data[index + 1]) - 1
        X = int(data[index + 2]) - 1
        constraints.append((L, R, X))
        index += 3
    
    # dp[i] will store the number of valid permutations of the first i elements
    dp = [0] * (N + 1)
    dp[0] = 1  # Base case: 1 way to arrange zero elements
    
    # max_not_at[i] will store the maximum value that cannot be at position i
    max_not_at = [None] * N
    
    # Process constraints
    for L, R, X in constraints:
        if max_not_at[X] is None:
            max_not_at[X] = R
        else:
            max_not_at[X] = max(max_not_at[X], R)
    
    # Prefix sums for fast range updates
    prefix = [0] * (N + 2)
    
    for i in range(1, N + 1):
        # Update prefix sums with the previous dp value
        prefix[1] += dp[i - 1]
        prefix[1] %= MOD
        
        # Calculate dp[i] using prefix sums
        dp[i] = (prefix[i] - prefix[i + 1]) % MOD
        
        # Apply constraints for the current position i-1
        if max_not_at[i - 1] is not None:
            R = max_not_at[i - 1]
            # We need to remove dp[i-1] from all dp[j] where j > R
            prefix[R + 2] -= dp[i - 1]
            prefix[R + 2] %= MOD
        
        # Update prefix sums for the next iteration
        for j in range(1, N + 2):
            prefix[j] += prefix[j - 1]
            prefix[j] %= MOD
    
    # The answer is the number of valid permutations of the first N elements
    print(dp[N])

if __name__ == "__main__":
    main()