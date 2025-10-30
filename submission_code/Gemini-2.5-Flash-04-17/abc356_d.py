import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    MOD = 998244353

    total_sum = 0

    # The problem asks for the sum for k from 0 to N.
    # popcount(k & M) = sum_{i=0 to infinity} (k_i & m_i).
    # The total sum is sum_{k=0 to N} sum_{i=0 to infinity} (k_i & m_i).
    # Swap summations: sum_{i=0 to infinity} sum_{k=0 to N} (k_i & m_i).
    # For a fixed bit position i, the inner sum is sum_{k=0 to N} (k_i & m_i).
    # If the i-th bit of M (m_i) is 0, then k_i & m_i = 0 for all k. The term for this i is 0.
    # If the i-th bit of M (m_i) is 1, then k_i & m_i = k_i. The inner sum is sum_{k=0 to N} k_i.
    # sum_{k=0 to N} k_i is the number of integers k in [0, N] such that the i-th bit of k is 1.
    # Let this count be C(N, i).
    # The total sum is sum_{i=0 to infinity, m_i=1} C(N, i).
    # Since M < 2^60, m_i = 0 for i >= 60. So we only need to sum for i from 0 to 59.

    # Function to compute C(N, i) mod MOD
    # C(N, i) = number of k in [0, N] such that the i-th bit of k is 1.
    # This count is given by the formula:
    # C(N, i) = floor((N+1) / 2^(i+1)) * 2^i + max(0, (N+1) mod 2^(i+1) - 2^i)
    def count_set_bits(N, i, MOD):
        N1 = N + 1
        pw2i = 1 << i
        pw2i1 = 1 << (i + 1)

        # Number of full blocks of size 2^(i+1) in [0, N] is q = (N+1) // 2^(i+1)
        q = N1 // pw2i1

        # The remainder size is r = (N+1) mod 2^(i+1)
        r = N1 % pw2i1

        # In each full block, there are 2^i numbers with the i-th bit set.
        # Contribution from full blocks: q * 2^i
        term1 = (q % MOD) * (pw2i % MOD) % MOD

        # In the partial block of size r, we count numbers x in [0, r-1] with the i-th bit set.
        # These are numbers x such that 2^i <= x < min(r, 2^(i+1)). Since x < r < 2^(i+1), this is 2^i <= x < r.
        # The number of such x is max(0, r - 2^i).
        partial_count = 0
        term2_raw = r - pw2i
        if term2_raw > 0:
            partial_count = term2_raw % MOD

        # Total count C(N, i) mod MOD is the sum of contributions mod MOD
        count_Ci = (term1 + partial_count) % MOD

        return count_Ci

    # Iterate through possible bit positions. Since M < 2^60, M_i is 0 for i >= 60.
    # Since N < 2^60, k <= N < 2^60, so k_i is 0 for i >= 60.
    # Thus, we only need to consider i from 0 up to 59.
    # A loop up to 60 covers i = 0, 1, ..., 59.
    for i in range(60):
        # Check if the i-th bit of M is set
        if (M >> i) & 1:
            # If M_i is 1, add C(N, i) to the total sum
            total_sum = (total_sum + count_set_bits(N, i, MOD)) % MOD

    print(total_sum)

solve()