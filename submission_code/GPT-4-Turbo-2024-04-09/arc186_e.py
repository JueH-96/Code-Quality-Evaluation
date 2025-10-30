MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    X = list(map(int, data[3:3+M]))
    
    # Total number of sequences of length N with elements from 1 to K
    total_sequences = pow(K, N, MOD)
    
    # We need to use dynamic programming to count sequences that can contain X as a subsequence
    # dp[i][j] will be the number of sequences of length i that can have the first j elements of X as a subsequence
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 1  # 1 way to have a sequence of length 0 that contains the first 0 elements of X (empty sequence)
    
    for i in range(1, N + 1):
        for j in range(M + 1):
            dp[i][j] = dp[i-1][j] * K % MOD  # All sequences of length i-1 extended by any of K elements
            if j > 0:
                # If we can use the j-th element of X to extend sequences of length i-1 that contain the first j-1 elements of X
                dp[i][j] = (dp[i][j] + dp[i-1][j-1] * (K - 1)) % MOD
    
    # The number of sequences of length N that can have X as a subsequence
    sequences_with_X = dp[N][M]
    
    # The number of valid sequences is total_sequences minus sequences_with_X
    result = (total_sequences - sequences_with_X + MOD) % MOD
    print(result)

if __name__ == "__main__":
    main()