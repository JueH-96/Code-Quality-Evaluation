def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    MOD = 998244353
    
    # Convert A from 1-based index to 0-based index
    A = [a - 1 for a in A]
    
    # Count how many times each index is referenced by A
    count = [0] * N
    for a in A:
        count[a] += 1
    
    # dp[i][j] will be the number of valid sequences where the first i elements
    # are filled and the i-th element has value exactly j.
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    
    # Initialize dp for the first element
    for j in range(1, M + 1):
        dp[1][j] = 1
    
    # Fill dp table
    for i in range(2, N + 1):
        # Accumulate sums for non-decreasing sequences
        cumulative = [0] * (M + 1)
        for j in range(1, M + 1):
            cumulative[j] = (cumulative[j - 1] + dp[i - 1][j]) % MOD
        
        for j in range(1, M + 1):
            # x_i = j, we need x_A[i-1] >= j
            a_index = A[i - 1]
            dp[i][j] = cumulative[j]
            if a_index != i - 1:
                dp[i][j] = (dp[i][j] * dp[i - 1][j]) % MOD
    
    # Sum all dp[N][j] for j = 1 to M to get the number of valid sequences
    result = sum(dp[N][j] for j in range(1, M + 1)) % MOD
    
    print(result)

if __name__ == "__main__":
    main()