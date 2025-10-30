def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    B = list(map(int, input_data[2:]))

    MOD = 998244353

    # ----------------------------------------------------------------
    # Key insight / formula:
    #
    # Let A be any full assignment of B (replacing -1's with [1..M]).
    # Define f(A) = number of "right-to-left maxima" in A, i.e.
    #   f(A) = count of indices i such that A[i] > max(A[i+1],...,A[N]),
    #   (with N-based indexing).
    #
    # We want SUM_{all expansions A} f(A) mod 998244353.
    #
    # We can rewrite:
    #   f(A) = sum_{i=1..N} I(i,A), where I(i,A)=1 if i is a R2L max, else 0.
    # So SUM_{A} f(A) = sum_{i=1..N} [number of expansions A for which i is R2L max].
    #
    # If i < N, for i to be R2L max we need A[i] > all of A[i+1..N].
    # If i = N, it is always a R2L max (there is nothing to its right),
    #   so as soon as A[N] is chosen/fixed, that index counts.
    #
    # Let T(i) = number of expansions where i is a R2L max.
    # Then answer = sum_{i=1..N} T(i).
    #
    # We can factor out the choices to the left of i (indices < i),
    # which do not affect whether A[i] > all to its right.
    #
    # Hence T(i) =  (number of ways to fill B[1..i-1])
    #             * (number of ways to fill B[i..N] so that A[i] > all A[j], j>i).
    #
    # The left block contributes M^(count_of_-1_in_[1..i-1]) (since each -1 there
    # can be anything from 1..M, and known values are fixed).
    #
    # The main task is: S(i) := # ways to fill subarray B[i..N] so that
    #   A[i] is strictly bigger than A[j] for all j>i.
    #
    # Case 1: If i=N, the condition is vacuously true (no j> N), so:
    #   - If B[N] != -1, there is exactly 1 way to fix A[N].
    #   - If B[N] == -1, there are M ways to choose A[N].
    #
    # Case 2: If i < N, let:
    #   max_known_from_right[i] = M_i = max( B[j] for j>i if B[j]!=-1 ), or 0 if none
    #   unknown_count_suffix[i+1] = number of -1's in B[i+1..N].
    #
    #   - If B[i] != -1 = w:
    #       we require w > all known B[j], j>i  =>  w > M_i
    #       also w must be > 1 if there is at least one j>i (because all B[j]>=1),
    #         otherwise it cannot be strictly bigger.
    #       then for each j>i with B[j] == -1, we must choose A[j] in [1..w-1],
    #         so that w > A[j]. That yields (w-1)^( number_of_-1_in_[i+1..N] ) ways.
    #       If w <= M_i or w <=1, then 0 ways.
    #
    #   - If B[i] == -1:
    #       we can pick A[i] = x in [1..M], but to be strictly bigger than
    #       all known B[j], j>i, we need x > M_i,
    #       also if there is at least one j>i, we need x>1 (since all A[j]>=1).
    #       So x must be >= 2 and x > M_i, i.e. x in [max(M_i+1, 2).. M].
    #       Then for each j>i with B[j] == -1, we have (x-1) choices for A[j].
    #       So # ways for a given x is (x-1)^(u_i), where u_i=# of -1 in (i+1..N).
    #       We sum over x in that range.
    #
    # Implementation plan:
    #
    # 1) Precompute powers and prefix sums of b^e mod for b=0..M, e=0..N,
    #    so we can quickly sum (x-1)^u_i from x=L..M in O(1).
    # 2) Compute unknown_count_prefix, unknown_count_suffix so we know how many
    #    -1 are to the left/right of each index i.
    # 3) Compute max_known_from_right[i] for i=1..N (the maximum known value in B[j], j>i).
    # 4) For each i in [1..N-1], compute S(i) by the cases above.
    #    For i=N, handle the trivial case.
    # 5) Then T(i) = M^(# -1 in [1..i-1]) * S(i).
    # 6) Sum T(i) over i=1..N, output result mod 998244353.
    #
    # Complexity:
    #  - Building the power tables prefix takes O(N*M).
    #  - Then computing T(i) for i=1..N is O(N).
    #  - For N,M up to 2000, O(N*M) = 4e6, which is on the edge but can be done
    #    in optimized Python (or typically in C++). We carefully implement it.
    #
    # ----------------------------------------------------------------

    # Count -1 to the left (prefix) and to the right (suffix).
    unknown_count_prefix = [0]*(N+1)  # unknown_count_prefix[i] = # of -1 in B[1..i]
    for i in range(1, N+1):
        unknown_count_prefix[i] = unknown_count_prefix[i-1] + (1 if B[i-1] == -1 else 0)

    unknown_count_suffix = [0]*(N+2)  # unknown_count_suffix[i] = # of -1 in B[i..N]
    for i in range(N, 0, -1):
        unknown_count_suffix[i] = unknown_count_suffix[i+1] + (1 if B[i-1] == -1 else 0)

    # max_known_from_right[i] = max known B[j] for j>i, or 0 if none
    max_known_from_right = [0]*(N+2)
    max_known_from_right[N+1] = 0
    for i in range(N, 0, -1):
        if i == N:
            max_known_from_right[i] = 0
        else:
            # check B[i] => i+1 in code's sense: B[i] is B_{i+1} in 1-based
            val = 0
            if B[i] != -1:
                val = B[i]
            max_known_from_right[i] = max(max_known_from_right[i+1], val)

    # Precompute M^k for k=0..N
    # (the number of ways to fill a block of k unknowns with any of 1..M)
    powM = [1]*(N+1)
    for e in range(1, N+1):
        powM[e] = (powM[e-1] * M) % MOD

    # ----------------------------------------------------------------
    # Build table pow_table_of[e][b] = b^e (mod), for e in [0..N], b in [0..M].
    # Then prefix_pow_of[e][b] = sum_{k=0..b} pow_table_of[e][k].
    # So we can sum (x-1)^e from x=L..M by summing k in [L-1..M-1].
    # Watch out for L-1 if L=1 or 2, etc.

    # pow_table_of[e][b] = b^e % MOD
    pow_table_of = [ [0]*(M+1) for _ in range(N+1) ]
    # e=0 => b^0=1 for any b>0, 0^0=1 by convention here
    for b in range(M+1):
        pow_table_of[0][b] = 1  # define 0^0 = 1 as well
    for e in range(1, N+1):
        # b=0 => 0^e=0 for e>0
        pow_table_of[e][0] = 0
        for b in range(1, M+1):
            pow_table_of[e][b] = (pow_table_of[e-1][b] * b) % MOD

    # Now prefix sums: prefix_pow_of[e][b] = sum_{k=0..b} pow_table_of[e][k]
    prefix_pow_of = [ [0]*(M+1) for _ in range(N+1) ]
    for e in range(N+1):
        running = 0
        for b in range(M+1):
            running = (running + pow_table_of[e][b]) % MOD
            prefix_pow_of[e][b] = running

    def range_sum(e, L, R):
        # sum of pow_table_of[e][k] for k in [L..R], inclusive
        if L>R: 
            return 0
        if L<=0:
            return prefix_pow_of[e][R]
        else:
            return (prefix_pow_of[e][R] - prefix_pow_of[e][L-1]) % MOD

    # ----------------------------------------------------------------
    # Now define a function S(i) = # ways to fill B[i..N] so that A[i] > all A[j], j>i.
    # We'll compute it on the fly inside the loop for i = 1..N-1,
    # and handle i=N as a special case.

    ans = 0

    for i in range(1, N+1):
        left_unknown = unknown_count_prefix[i-1]  # # -1 in B[1..i-1]
        ways_left = powM[left_unknown]  # ways to assign left part

        if i == N:
            # S(N): trivially i = N is a R2L max
            if B[N-1] != -1:
                # exactly 1 way to fix B[N]
                s_val = 1
            else:
                # B[N] = -1 => can choose any in [1..M]
                s_val = M
        else:
            # i < N
            M_i = max_known_from_right[i]  # max known to the right
            u_i = unknown_count_suffix[i+1]  # #unknowns in B[i+1..N]
            bi = B[i-1]

            if bi != -1:
                # known value = w
                w = bi
                # for i<N, we need w> M_i and w>1 to be strictly bigger than all B[j],
                #  j> i (which are >=1). If w <= M_i or w<=1 => 0
                if w <= M_i or w <= 1:
                    s_val = 0
                else:
                    # (w-1)^u_i
                    s_val = pow_table_of[u_i][w-1]  # (w-1)^{u_i}
            else:
                # bi == -1
                # x must be in [max(M_i+1, 2) .. M]
                L = max(M_i+1, 2)
                if L> M:
                    s_val = 0
                else:
                    # sum_{x=L..M} (x-1)^{u_i} = sum_{k=L-1..M-1} pow_table_of[u_i][k]
                    # where k = x-1
                    s_val = range_sum(u_i, L-1, M-1)

        T_i = (ways_left * s_val) % MOD
        ans = (ans + T_i) % MOD

    print(ans % MOD)