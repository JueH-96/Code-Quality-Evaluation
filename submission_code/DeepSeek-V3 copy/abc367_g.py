MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    A = list(map(int, data[3:3+N]))
    
    # Precompute the number of subsequences of length m for each m
    # We need to find the number of subsequences of length m for each m that is a multiple of M
    # The total number of subsequences of length m is C(N, m)
    # We need to sum over all m that are multiples of M
    
    # Precompute factorials and inverse factorials modulo MOD
    fact = [1] * (N + 1)
    inv_fact = [1] * (N + 1)
    
    for i in range(1, N+1):
        fact[i] = fact[i-1] * i % MOD
    
    inv_fact[N] = pow(fact[N], MOD-2, MOD)
    for i in range(N-1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
    
    def comb(n, k):
        if k < 0 or k > n:
            return 0
        return fact[n] * inv_fact[k] % MOD * inv_fact[n-k] % MOD
    
    # Now, for each m that is a multiple of M, we need to compute the sum of (xor of the subsequence)^K over all subsequences of length m
    # To compute this, we can use dynamic programming
    # Let dp[m][x] be the number of subsequences of length m with xor x
    # Initialize dp[0][0] = 1
    # For each element a in A, update dp[m][x] += dp[m-1][x ^ a]
    
    # Since M is small (up to 100), we can iterate over all possible m that are multiples of M
    # The maximum m is N
    
    total = 0
    
    for m in range(M, N+1, M):
        # Initialize dp
        dp = [dict() for _ in range(m+1)]
        dp[0][0] = 1
        
        for a in A:
            new_dp = [dict() for _ in range(m+1)]
            for i in range(m+1):
                for x, cnt in dp[i].items():
                    # Option 1: not take a
                    if x in new_dp[i]:
                        new_dp[i][x] = (new_dp[i][x] + cnt) % MOD
                    else:
                        new_dp[i][x] = cnt % MOD
                    # Option 2: take a, if i < m
                    if i < m:
                        y = x ^ a
                        if y in new_dp[i+1]:
                            new_dp[i+1][y] = (new_dp[i+1][y] + cnt) % MOD
                        else:
                            new_dp[i+1][y] = cnt % MOD
            dp = new_dp
        
        # Now, sum over all x in dp[m], (x)^K * count
        for x, cnt in dp[m].items():
            total = (total + pow(x, K, MOD) * cnt) % MOD
    
    print(total)

if __name__ == "__main__":
    main()