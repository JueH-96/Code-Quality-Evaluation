import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1000000)
    mod = 998244353

    N, M, K = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    # Precompute Stirling numbers of the second kind S[n][k] up to N,K
    # S[n][k] = k * S[n-1][k] + S[n-1][k-1]
    S = [[0] * (K+1) for _ in range(N+1)]
    S[0][0] = 1
    for n in range(1, N+1):
        for k in range(1, min(n, K) + 1):
            S[n][k] = (k * S[n-1][k] + S[n-1][k-1]) % mod

    # factorial
    fact = [1] * (K+1)
    for i in range(1, K+1):
        fact[i] = fact[i-1] * i % mod

    # Precompute block contributions:
    # For s in [0..M-2], block uses f[t] = t * (K-1)! * S[t-1][K-1] for t>=K else 0
    # For s = M-1, block uses g[t] = (K-1)! * S[t][K-1] for t>=K-1 else 0
    maxT = N
    f = [0] * (maxT+1)
    for t in range(K, maxT+1):
        # loops length t-1 must map onto (K-1) symbols
        onto = S[t-1][K-1] * fact[K-1] % mod
        f[t] = t * onto % mod
    g = [0] * (maxT+1)
    for t in range(K-1, maxT+1):
        onto = S[t][K-1] * fact[K-1] % mod
        g[t] = onto

    # DP over blocks 0..M-1
    # dp[i][j] = ways to fill first i blocks summing to j
    dp = [ [0] * (N+1) for _ in range(M+1) ]
    dp[0][0] = 1
    for i in range(1, M+1):
        # block index = i-1
        is_last = (i == M)
        low = K-1 if is_last else K
        arr = g if is_last else f
        # convolution dp[i-1] with arr
        for used in range(0, N+1):
            v = dp[i-1][used]
            if v == 0: continue
            # add block of size t
            # t from low to N-used
            upto = N - used
            for t in range(low, upto+1):
                c = arr[t]
                if c:
                    dp[i][used + t] = (dp[i][used + t] + v * c) % mod

    ans = dp[M][N]
    print(ans)

if __name__ == "__main__":
    main()