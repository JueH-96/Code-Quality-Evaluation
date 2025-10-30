def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MOD = 998244353

    # ----------------------------
    # 1) Check boundary feasibility:
    #
    #   The problem's cells are 1-indexed.
    #   Initial X[1] = 1, X[2] = 0, X[3]=1,... i.e. X[i] = i % 2.
    #   We can never change the bit in cell 1 or in cell N, so:
    #       A[1] must equal X[1] = 1
    #       A[N] must equal X[N] = (N % 2)
    #
    #   In 0-based Python indexing:
    #       A[0] must be 1
    #       A[N-1] must be (N % 2)
    # ----------------------------

    if A[0] != 1:
        print(0)
        return
    if (N % 2) == 0:
        # N even => X[N] = 0
        if A[N-1] != 0:
            print(0)
            return
    else:
        # N odd => X[N] = 1
        if A[N-1] != 1:
            print(0)
            return

    # ----------------------------
    # 2) Build the set of "final transitions" T_f.
    #    A transition at boundary i (1<=i<N) means A[i] != A[i+1].
    #
    #    In 0-based code, "boundary i" will be i where 0<=i<N-1,
    #    meaning the boundary between A[i] and A[i+1].
    #
    #    We'll store these boundary indices 1..N-1 in 1-based style,
    #    but implement with i in [1..N-1].
    # ----------------------------

    # T_f will be stored as a list of integers in [1..N-1].
    T_f = []
    for i in range(N-1):
        if A[i] != A[i+1]:
            T_f.append(i+1)  # +1 to make it 1-based boundary

    # ----------------------------
    # 3) We want to remove all boundaries in S = {1..N-1} \ T_f.
    #    But per the rules, each operation removes exactly two
    #    "still-present" boundaries that are neighbors in the current set T,
    #    provided there is no other boundary of T between them.
    #
    #    Equivalently, if we list T_f (plus 0 and N as sentinels),
    #    the open intervals between consecutive elements of T_f
    #    contain those boundaries we must remove.  In each such interval
    #    we end up removing all of them pairwise in a "dynamic adjacency"
    #    fashion.  A known result is:
    #
    #      • An interval of length M (even) can be removed in (M-1)!! ways,
    #        where (2k-1)!! = 1*3*5*...(2k-1).
    #        Equivalently (2k-1)!! = (2k)! / (2^k * k!).
    #
    #      • If we have c disjoint intervals, each with M_i boundaries,
    #        then after computing the number of ways in each interval,
    #        we still must count how to interleave these merges among
    #        the c intervals.  Each interval i needs M_i/2 merges.
    #        The total merges = sum(M_i/2).  The number of ways to
    #        interleave c groups of merges of sizes L_i = M_i/2 is
    #        (L_1+...+L_c)! / (L_1! * ... * L_c!).
    #
    #    So final answer =
    #      ( product over intervals of (M_i-1)!! ) *
    #      ( factorial( sum(L_i) ) / product( factorial(L_i) ) ).
    #
    # ----------------------------

    # Make a sorted list B = [0] + T_f + [N].
    # Then intervals are (B[j], B[j+1]) => boundaries from B[j]+1 .. B[j+1]-1
    # length of each interval M_j = B[j+1] - B[j] - 1
    # those are the boundaries in S in that range.

    B = [0] + T_f + [N]
    # Precompute factorials, inverses, powers of 2 up to 2*N (to be safe)
    # because we might need up to N-1 boundaries.
    maxF = 2*N + 2

    fact = [1]*(maxF)
    invfact = [1]*(maxF)
    for i in range(1, maxF):
        fact[i] = fact[i-1]*i % MOD

    # Fermat-inverse via fact[n]^(MOD-2) mod:
    invfact[maxF-1] = pow(fact[maxF-1], MOD-2, MOD)
    for i in reversed(range(maxF-1)):
        invfact[i] = invfact[i+1]*(i+1) % MOD

    # Precompute 2^k and inverse of 2^k up to N
    # but let's just do a function for "inv(2^k)" on the fly or store it
    # We'll store them for speed anyway.
    maxPow2 = N+2
    pow2 = [1]* (maxPow2)
    inv2 = [1]* (maxPow2)
    for i in range(1, maxPow2):
        pow2[i] = (pow2[i-1]*2) % MOD
    inv2[maxPow2-1] = pow(pow2[maxPow2-1], MOD-2, MOD)
    for i in reversed(range(maxPow2-1)):
        inv2[i] = (inv2[i+1]*2) % MOD

    def ways_to_remove_even(M):
        """
        If M=2k, the number of ways to remove those M boundaries in a single
        interval (dynamic adjacency) is (2k-1)!! = (2k)! / (2^k * k!).
        If M=0, we take it as 1 (nothing to remove).
        """
        if M == 0:
            return 1
        # M is even by assumption
        k = M//2
        # (2k)! / (2^k * k!)
        numerator = fact[M]               # (2k)!
        denominator = (pow2[k] * fact[k]) % MOD  # 2^k * k!
        return (numerator * pow(denominator, MOD-2, MOD)) % MOD

    ways_segments = 1
    total_merges = 0

    # Collect merges_i to use in the interleaving factorial
    merges_list = []

    for j in range(len(B)-1):
        left = B[j]
        right = B[j+1]
        M = right - left - 1   # number of boundaries in (left,right)
        if M < 0:
            continue  # no interval
        if M == 0:
            merges_list.append(0)
            continue
        # M > 0 => must be removable => must be even or else impossible
        if (M % 2) != 0:
            print(0)
            return
        # number of ways in this interval
        ways_int = ways_to_remove_even(M)
        ways_segments = (ways_segments * ways_int) % MOD
        merges_i = M//2
        merges_list.append(merges_i)
        total_merges += merges_i

    # Now compute the ways to interleave
    #   ways_interleave = fact(total_merges) / Π( fact( merges_i ) )
    #   but be careful with mod.

    ways_interleave = fact[total_merges]
    for m_i in merges_list:
        ways_interleave = (ways_interleave * invfact[m_i]) % MOD

    answer = (ways_segments * ways_interleave) % MOD
    print(answer)

# Do not forget to call main!
main()