import sys

# Set recursion depth limit if necessary (usually not needed for this problem structure)
# sys.setrecursionlimit(2000)

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

MOD = 998244353

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    A_in = list(map(int, sys.stdin.readline().split()))
    # A_in[i] corresponds to A_{i+2}

    A_val = [0] * (N + 1) # A_val[i] will store A_i for i=2...N
    for i in range(len(A_in)):
        A_val[i + 2] = A_in[i]

    fact = [1] * N
    for i in range(1, N):
        fact[i] = (fact[i - 1] * i) % MOD

    inv = [0] * (N + 1)
    # inv[i] for i >= 1
    for i in range(1, N + 1):
        inv[i] = modInverse(i)

    # Prefix sums S'_i = sum_{k=2}^{i} A_k * inv[k] mod MOD
    # S'_i stores sum from k=2 up to index i
    S_prime = [0] * (N + 1)
    for i in range(2, N + 1):
        S_prime[i] = (S_prime[i - 1] + A_val[i] * inv[i]) % MOD

    fact_N_minus_1 = fact[N - 1]

    for _ in range(Q):
        u, v = map(int, sys.stdin.readline().split())
        # The problem guarantees u_i < v_i, so no need to swap
        # if u > v:
        #     u, v = v, u

        ans = 0
        if u == 1:
            # Formula for u=1, v>1
            # Sum_{k=2}^{v-1} A_k/k + A_v
            # = S'_{v-1} + A_v
            # Note: if v=2, the sum Sum_{k=2}^{v-1} is empty (k goes from 2 to 1). S'_{2-1}=S'_1=0. Correct.
            term_sum_k_2_v_minus_1 = S_prime[v - 1] # S'_{v-1}
            term_Av = A_val[v]
            
            inner_sum = (term_sum_k_2_v_minus_1 + term_Av) % MOD

            ans = (fact_N_minus_1 * inner_sum) % MOD
        else:
            # Formula for 1 < u < v
            # Sum = (N-1)! * [ Sum_{k=2}^{u-1} A_k/k + A_u * (u-1)/u + Sum_{k=u+1}^{v-1} A_k/k + A_v ]
            # Sum_{k=2}^{u-1} A_k/k = S'_{u-1}
            term1 = S_prime[u - 1] # S'_{u-1}

            # A_u * (u-1)/u = A_u * (u-1) * inv[u]
            term_Au_contrib = (A_val[u] * (u - 1) * inv[u]) % MOD

            # Sum_{k=u+1}^{v-1} A_k/k = S'_{v-1} - S'_u
            term3 = (S_prime[v - 1] - S_prime[u] + MOD) % MOD # S'_{v-1} - S'_u

            # A_v term
            term4 = A_val[v]

            inner_sum = (term1 + term_Au_contrib + term3 + term4) % MOD
            ans = (fact_N_minus_1 * inner_sum) % MOD

        print(ans)

solve()