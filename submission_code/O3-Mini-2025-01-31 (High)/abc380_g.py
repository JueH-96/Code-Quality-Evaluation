# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    try:
        N = int(next(it))
        K = int(next(it))
    except StopIteration:
        return
    N = int(N)
    K = int(K)
    # Use 1-indexing for convenience.
    P = [0]*(N+1)
    for i in range(1, N+1):
        P[i] = int(next(it))
    mod = 998244353

    # === Compute T, the number of inversions in the entire permutation ===
    size = N
    bitT = [0]*(size+1)
    # BIT update and query:
    def bitT_update(i, delta):
        while i <= size:
            bitT[i] += delta
            i += i & -i
    def bitT_query(i):
        s_ = 0
        while i:
            s_ += bitT[i]
            i -= i & -i
        return s_
    T_val = 0
    for i in range(1, N+1):
        x = P[i]
        # count how many of the already–processed numbers are ≤ x
        cnt = bitT_query(x)
        # then (i-1 - cnt) numbers (before i) are greater than x
        T_val += (i - 1 - cnt)
        bitT_update(x, 1)
    # T_val now is the inversion count of the whole permutation.

    # === Compute S = sum_{s=1}^{M} (inversion count in block P[s..s+K-1]) ===
    M_val = N - K + 1  # number of possible segments
    # We wish to “sum” over all windows the inversion–count internal to the window.
    # A standard observation shows that if we define for each inversion pair (i,j) (with i<j)
    # a weight L(i,j) = max(0, min(i, M_val) - max(j-K+1, 1) + 1),
    # then S = Σ_{i<j, P[i] > P[j]} L(i,j).
    # We compute S by “sweeping” i from 1 to N while maintaining as an active set all indices j
    # that lie in the current window [i+1, min(i+K-1, N)].
    # In the current window (for fixed i) we add for every j with P[j] < P[i] a contribution
    # = (min(i, M_val)+1 - max(j-K+1,1)).
    # Notice: max(j-K+1,1) is simply:
    #   1 for j <= K, and j-K+1 for j > K.
    
    # We maintain two BITs keyed by the _value_ (not the index) of P:
    size2 = N
    bit_count = [0]*(size2+1)
    bit_sum = [0]*(size2+1)
    def bit_update(bit, i, delta):
        while i <= size2:
            bit[i] += delta
            i += i & -i
    def bit_query(bit, i):
        s_ = 0
        while i:
            s_ += bit[i]
            i -= i & -i
        return s_
    # Precompute array B where for each index j, B[j] = 1 if j<=K else (j-K+1)
    B = [0]*(N+1)
    for j in range(1, N+1):
        if j <= K:
            B[j] = 1
        else:
            B[j] = j - K + 1

    # For each i the "active set" will be indices j with j in [i+1, min(i+K-1, N)].
    # Initialize active set for i=1:
    lo = 2
    hi = K if K <= N else N
    for j in range(lo, hi+1):
        bit_update(bit_count, P[j], 1)
        bit_update(bit_sum, P[j], B[j])
    S_val = 0
    for i in range(1, N+1):
        # For fixed i, define A = min(i, M_val) + 1.
        A_val = (i if i < M_val else M_val) + 1
        # Query active set for all j with P[j] < P[i]
        cnt = bit_query(bit_count, P[i] - 1)
        sB = bit_query(bit_sum, P[i] - 1)
        S_val += A_val * cnt - sB
        # Slide the active set: for next i, remove j = i+1 and add j = i+K.
        if K > 1:  # only non-trivial when window length >= 1
            rem = i + 1
            if rem <= N and rem <= i + K - 1:
                bit_update(bit_count, P[rem], -1)
                bit_update(bit_sum, P[rem], -B[rem])
            add = i + K
            if add <= N:
                bit_update(bit_count, P[add], 1)
                bit_update(bit_sum, P[add], B[add])
    # Now S_val = sum_{s=1}^{M_val} (inversion count inside the block P[s...s+K-1]).

    # === Final answer ===
    # Our analysis shows that the expected inversion count after the operation is
    #   E = T + (K*(K-1))/4 - (S_val / M_val)
    # We must output E modulo 998244353; note that (K*(K-1))/4 and division by M_val are done modulo mod.
    inv4 = pow(4, mod-2, mod)
    invM = pow(M_val, mod-2, mod)
    term1 = T_val % mod
    term2 = (K * (K - 1)) % mod
    term2 = (term2 * inv4) % mod
    term3 = (S_val % mod) * invM % mod
    ans = (term1 + term2 - term3) % mod
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()