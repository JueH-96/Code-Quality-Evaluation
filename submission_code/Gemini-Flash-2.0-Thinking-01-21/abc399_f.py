import sys

def solve():
    input = sys.stdin.readline
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    MOD = 998244353
    
    # 1. Compute prefix sums P[i] = sum_{j=1 to i} A_j mod MOD
    # P[0] = 0
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = (P[i] + A[i]) % MOD
    
    # 2. Precompute powers
    # pow_P[i][j] stores P[i]^j mod MOD for i=0..N, j=0..K
    # Used for Q calculation, needs P[1] to P[N]
    pow_P = [[0] * (K + 1) for _ in range(N + 1)]
    for i in range(N + 1): # P[0]..P[N]
        pow_P[i][0] = 1
        for j in range(1, K + 1):
            pow_P[i][j] = (pow_P[i][j-1] * P[i]) % MOD
            
    # neg_pow_P[i][k] stores ((-P[i]) mod MOD)^k mod MOD for i=0..N-1, k=0..K
    # Used for T calculation, needs P[0] to P[N-1]
    neg_pow_P = [[0] * (K + 1) for _ in range(N)]
    for i in range(N): # P[0]..P[N-1]
        neg_base = (MOD - P[i]) % MOD
        neg_pow_P[i][0] = 1
        for k in range(1, K + 1):
            neg_pow_P[i][k] = (neg_pow_P[i][k-1] * neg_base) % MOD

    # 3. Compute suffix sums of powers Q[j][m]
    # Q[j][m] stores sum_{k=m to N} P[k]^j mod MOD for j=0..K, m=1..N+1
    # Q[j][N+1] = 0
    # Needs pow_P[k][j] for k=m..N
    Q = [[0] * (N + 2) for _ in range(K + 1)]
    
    for j in range(K + 1):
        for m in range(N, 0, -1): # m from N down to 1
            Q[j][m] = (Q[j][m+1] + pow_P[m][j]) % MOD

    # 4. Compute T[j] = sum_{i=0 to N-1} (-P[i])^(K-j) * Q[j][i+1] mod MOD for j=0..K
    # Needs neg_pow_P[i][K-j] for i=0..N-1
    # Needs Q[j][i+1] where i+1 goes from 1 to N.
    T = [0] * (K + 1)
    for j in range(K + 1):
        for i in range(N): # i goes from 0 to N-1
            # term = neg_pow_P[i][K-j] * Q[j][i+1]
            term = (neg_pow_P[i][K-j] * Q[j][i+1]) % MOD
            T[j] = (T[j] + term) % MOD

    # 5. Compute binomial coefficients C(K, j) for j=0..K
    C = [[0] * (K + 1) for _ in range(K + 1)]
    for i in range(K + 1):
        C[i][0] = 1
        C[i][i] = 1
        for j in range(1, i):
            C[i][j] = (C[i-1][j-1] + C[i-1][j]) % MOD

    # 6. Compute final sum = sum_{j=0 to K} C(K, j) * T[j] mod MOD
    total_sum = 0
    for j in range(K + 1):
        term = (C[K][j] * T[j]) % MOD
        total_sum = (total_sum + term) % MOD
    
    print(total_sum)

solve()