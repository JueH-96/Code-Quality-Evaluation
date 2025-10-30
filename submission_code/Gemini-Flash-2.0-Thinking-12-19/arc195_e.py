import sys

def solve():
    MOD = 998244353

    def power(a, b):
        res = 1
        a %= MOD
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % MOD
            a = (a * a) % MOD
            b //= 2
        return res

    def modInverse(n):
        return power(n, MOD - 2)

    # Read N and Q
    N, Q = map(int, sys.stdin.readline().split())

    # Precomputation
    # fact[i] stores i! mod MOD
    fact = [1] * (N + 1)
    for i in range(2, N + 1):
        fact[i] = (fact[i-1] * i) % MOD

    # inv[i] stores modular inverse of i mod MOD
    # inv_sq[i] stores modular inverse of i^2 mod MOD
    inv = [1] * (N + 1)
    inv_sq = [1] * (N + 1)
    
    # Compute inv[N] using Fermat's Little Theorem
    if N >= 1: # N >= 2 based on constraints, but >=1 for completeness
        inv[N] = modInverse(N)
        inv_sq[N] = (inv[N] * inv[N]) % MOD

    # Compute inv[i] and inv_sq[i] for i = N-1 down to 1
    # inv[i] = inv[i+1] * (i+1) mod MOD
    # inv_sq[i] = inv_sq[i+1] * (i+1)^2 mod MOD
    for i in range(N - 1, 1, -1):
        inv[i] = (inv[i+1] * (i+1)) % MOD
        inv_sq[i] = (inv_sq[i+1] * (i+1) % MOD * (i+1) % MOD) % MOD 

    # inv[1] and inv_sq[1] are 1, initialized correctly

    # Read A_i values. A[i] stores the weight for edge (i, P_i) for i=2..N
    # Input gives A_2, A_3, ..., A_N
    # A list size N+1. A[0], A[1] unused. A[2]..A[N] stored.
    A = [0] * (N + 1)
    A_input = list(map(int, sys.stdin.readline().split()))
    for i in range(2, N + 1):
        A[i] = A_input[i-2]

    # Precompute prefix sums S1 and S2
    # S1[k] = sum_{i=2..k} A[i] * inv[i] mod MOD
    # S2[k] = sum_{i=2..k} A[i] * inv_sq[i] mod MOD
    S1 = [0] * (N + 1)
    S2 = [0] * (N + 1)
    
    for i in range(2, N + 1):
        S1[i] = (S1[i-1] + (A[i] * inv[i]) % MOD) % MOD
        S2[i] = (S2[i-1] + (A[i] * inv_sq[i]) % MOD) % MOD

    # (N-1)! mod MOD
    factN1 = fact[N-1]

    # Process queries
    for _ in range(Q):
        u, v = map(int, sys.stdin.readline().split())
        # Ensure u < v
        if u > v:
            u, v = v, u

        # Calculate the sum of coefficients for FactN1
        # S(u,v) = FactN1 * ( sum_{i=2..u-1} A[i]*2*(i-1)*i^{-2} + A[u]*(u-1)*u^{-1} (if u>=2) + sum_{i=u+1..v-1} A[i]*i^{-1} + A[v] )
        
        # Term 1: Sum_{i=2 to u-1} A[i] * 2 * (i-1) * inv[i]^2
        # = 2 * Sum_{i=2 to u-1} A[i] * (i * inv[i]^2 - inv[i]^2)
        # = 2 * Sum_{i=2 to u-1} A[i] * (inv[i] - inv_sq[i])
        # = 2 * (S1[u-1] - S2[u-1])
        term1 = (2 * (S1[u-1] - S2[u-1] + MOD)) % MOD

        # Term 2: A[u] * (u-1) * inv[u] (coefficient for i=u)
        # Only applicable if u >= 2
        term2 = 0
        if u >= 2:
            term2 = (A[u] * (u-1) * inv[u]) % MOD

        # Term 3: Sum_{i=u+1 to v-1} A[i] * inv[i]
        # = S1[v-1] - S1[u]
        term3 = (S1[v-1] - S1[u] + MOD) % MOD

        # Term 4: A[v] (coefficient for i=v)
        term4 = A[v]

        # Total sum of coefficients for FactN1
        total_sum_coeffs = (term1 + term2 + term3 + term4) % MOD
        
        # Final answer = (N-1)! * total_sum_coeffs mod MOD
        ans = (factN1 * total_sum_coeffs) % MOD
        sys.stdout.write(str(ans) + '
')

solve()