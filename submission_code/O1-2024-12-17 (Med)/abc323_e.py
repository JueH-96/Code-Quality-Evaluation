def main():
    import sys
    input = sys.stdin.readline
    MOD = 998244353
    
    # Read inputs
    N, X = map(int, input().split())
    T = list(map(int, input().split()))
    T1 = T[0]
    
    # Modular inverse of N
    invN = pow(N, MOD - 2, MOD)
    
    # DP array where dp[t] = probability (mod MOD) that
    # we are exactly at a song boundary at time t
    dp = [0] * (X + 1)
    dp[0] = 1  # at time 0, we are certainly at a boundary
    
    # Populate dp using the fact that from boundary t,
    # we choose one of the N songs (each with prob 1/N),
    # so dp[t + T_i] += dp[t]*(1/N), if t + T_i <= X
    for t in range(X + 1):
        if dp[t] != 0:
            p = (dp[t] * invN) % MOD
            for length in T:
                nxt = t + length
                if nxt <= X:
                    dp[nxt] = (dp[nxt] + p) % MOD
    
    # We want the probability that song 1 is playing at time X + 0.5.
    # Song 1 will be playing at time X+0.5 if we started song 1 at some
    # boundary t (0 <= t <= X) and t + T1 > X + 0.5.  Since T1,t are integer,
    # this is equivalent to t + T1 >= X + 1.
    # Probability of picking song 1 at each boundary is 1/N.
    # So we sum dp[t]*(1/N) over all t where t + T1 >= X+1.
    start_t = X + 1 - T1
    if start_t < 0:
        start_t = 0
    
    ans = 0
    for t in range(start_t, X + 1):
        ans = (ans + dp[t] * invN) % MOD
    
    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()