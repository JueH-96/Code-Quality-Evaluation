import sys

def solve():
    """
    Solves the problem by calculating the sum of connected components over all possible sequences.
    """
    try:
        # Fast I/O
        input = sys.stdin.readline
        N, M = map(int, input().split())
        B_input = list(map(int, input().split()))
    except (IOError, ValueError):
        # Handle cases with no input
        return

    # Use 1-based indexing for B for easier implementation of logic
    B = [0] + B_input

    MOD = 998244353

    q_total = B.count(-1)

    # Precompute powers: pow_t[t][k] = t^k, pow_M_minus_t[t][k] = (M-t)^k
    pow_t = [[0] * (N + 1) for _ in range(M + 1)]
    pow_M_minus_t = [[0] * (N + 1) for _ in range(M + 1)]

    for t in range(M + 1):
        # Powers of t
        if t > 0:
            pow_t[t][0] = 1
            for k in range(1, N + 1):
                pow_t[t][k] = (pow_t[t][k - 1] * t) % MOD
        
        # Powers of M-t
        val_M_minus_t = M - t
        if val_M_minus_t >= 0:
            pow_M_minus_t[t][0] = 1
            for k in range(1, N + 1):
                pow_M_minus_t[t][k] = (pow_M_minus_t[t][k - 1] * val_M_minus_t) % MOD

    # Contribution from i = 1
    # For i=1, the condition `min_{u<1} A_u > max_{v>=1} A_v` is always true.
    # The number of ways to fill -1s is M^q.
    total_sum = pow(M, q_total, MOD)
    
    # Precompute prefix/suffix properties for the given sequence B
    pref_q = [0] * (N + 1)
    for i in range(1, N + 1):
        pref_q[i] = pref_q[i-1] + (1 if B[i] == -1 else 0)

    # pref_min_B[i] = min over non-(-1) values in B[1...i]
    pref_min_B = [M + 1] * (N + 2)
    for i in range(1, N + 1):
        pref_min_B[i] = pref_min_B[i-1]
        if B[i] != -1:
            pref_min_B[i] = min(pref_min_B[i], B[i])

    # suff_max_B[i] = max over non-(-1) values in B[i...N]
    suff_max_B = [0] * (N + 2)
    for i in range(N, 0, -1):
        suff_max_B[i] = suff_max_B[i+1]
        if B[i] != -1:
            suff_max_B[i] = max(suff_max_B[i], B[i])

    # Calculate contributions for i = 2 to N
    for i in range(2, N + 1):
        q1 = pref_q[i-1]
        q2 = q_total - q1
        
        min_b1 = pref_min_B[i-1]
        max_b2 = suff_max_B[i]

        # The condition on fixed values restricts the possible range of threshold t.
        # min_b1 > t  => t <= min_b1 - 1
        # max_b2 <= t => t >= max_b2
        # So we sum over t where max_b2 <= t <= min_b1 - 1.
        L = max_b2
        R = min_b1 - 1

        if L > R:
            continue
        
        i_sum = 0
        # The threshold t must be an integer in [1, M-1].
        start_t = max(1, L)
        end_t = min(M - 1, R)
        
        for t in range(start_t, end_t + 1):
            term = (pow_M_minus_t[t][q1] * pow_t[t][q2]) % MOD
            i_sum = (i_sum + term) % MOD
        
        total_sum = (total_sum + i_sum) % MOD

    print(total_sum)

solve()