def main():
    MOD = 998244353
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    coords = list(map(int, N_and_rest[1:]))
    X = coords[::2]
    Y = coords[1::2]
    
    # Sort balls by X
    balls = sorted(zip(X, Y), key=lambda x: x[0])
    Y_sorted = [ball[1] for ball in balls]
    
    # Initialize dp table
    dp = [[0] * (N + 1) for _ in range(N + 2)]
    for k in range(N + 1):
        dp[N+1][k] = 1  # Base case
    
    # Fill dp table from N down to 1
    for i in range(N, 0, -1):
        Y_i = Y_sorted[i-1]
        for k in range(N + 1):
            dp[i][k] = dp[i+1][k]  # Exclude ball i
            if Y_i <= k:
                dp[i][k] = (dp[i][k] + dp[i+1][Y_i]) % MOD  # Include ball i
    
    # The answer is dp[1][N]
    print(dp[1][N] % MOD)

if __name__ == '__main__':
    main()