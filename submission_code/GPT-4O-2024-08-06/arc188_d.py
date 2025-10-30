# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    B = list(map(int, data[N+1:2*N+1]))
    
    # Initialize the DP table
    dp = [[0] * (2 * N + 1) for _ in range(N + 1)]
    dp[0][0] = 1
    
    # Precompute factorials and inverse factorials for combinations
    fact = [1] * (2 * N + 1)
    inv_fact = [1] * (2 * N + 1)
    
    def modinv(x, mod):
        return pow(x, mod - 2, mod)
    
    for i in range(2, 2 * N + 1):
        fact[i] = fact[i - 1] * i % MOD
    
    inv_fact[2 * N] = modinv(fact[2 * N], MOD)
    for i in range(2 * N - 1, 0, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
    
    # Count the number of valid sequences
    for i in range(N):
        for j in range(2 * N + 1):
            if dp[i][j] == 0:
                continue
            # Try to place the next sequence
            for k in range(1, N + 1):
                if k not in A[:i] and k not in B[:i]:
                    # Calculate the new position
                    new_pos = j + 1
                    if new_pos <= 2 * N:
                        dp[i + 1][new_pos] = (dp[i + 1][new_pos] + dp[i][j]) % MOD
    
    # Calculate the result
    result = 0
    for j in range(2 * N + 1):
        result = (result + dp[N][j]) % MOD
    
    print(result)