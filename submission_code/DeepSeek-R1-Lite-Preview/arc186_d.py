MOD = 998244353

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    
    # Precompute factorial and inverse factorial for Catalan numbers
    MAX = 2 * N + 10
    factorial = [1] * (MAX)
    for i in range(1, MAX):
        factorial[i] = factorial[i-1] * i % MOD
    inv_fact = [1] * (MAX)
    inv_fact[MAX-1] = pow(factorial[MAX-1], MOD-2, MOD)
    for i in range(MAX-2, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
    
    # Function to compute Catalan number C(n)
    def catalan(n):
        if n == 0:
            return 1
        return factorial[2*n-2] * inv_fact[n-1] % MOD * inv_fact[n] % MOD
    
    # Precompute dp[N] = C(N-1)
    dp_N = catalan(N-1)
    
    # Now, handle the lex order constraint with A
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def dp(pos, cnt, tight):
        if pos == N:
            return 1 if cnt == 0 else 0
        res = 0
        max_val = A[pos] if tight else N-1
        for v in range(0, max_val+1):
            new_tight = tight and (v == A[pos])
            if cnt - 1 + v < 0:
                continue
            res = (res + dp(pos+1, cnt - 1 + v, new_tight)) % MOD
        return res
    
    result = dp(0, A[0], True)
    print(result % MOD)

if __name__ == '__main__':
    main()