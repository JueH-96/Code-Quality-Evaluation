def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S_str = input_data[1]
    C = list(map(int, input_data[2:]))

    # Convert the string S of '0'/'1' to a list of integers 0/1 for easier handling
    S = [0 if ch == '0' else 1 for ch in S_str]

    # We want to form a new string T (also length N of 0/1) so that:
    #  - Exactly one pair of adjacent bits in T is the same (T[i] == T[i+1]),
    #    and all other adjacent pairs are different.
    #
    # Equivalently, T must alternate everywhere except for exactly one "break."
    # That is, T looks like ...0,1,0,1... except at exactly one index i (1 <= i < N)
    # where T[i] = T[i+1]. 
    #
    # We can flip each S[i] to 1 - S[i] at cost C[i]. We want to minimize total flip cost.
    #
    # ------------------------------------------------------------
    # STRATEGY:
    #
    # 1) For each i in [1..N-1], we decide that the "break" (the pair that is the same)
    #    happens between positions i and i+1.
    # 2) We also decide an alternating pattern for T[1..i], which can start either with 0 or with 1.
    #    Let p in {0,1} be the starting bit for T_1, so T_1 = p, T_2 = 1-p, T_3 = p, etc., up to T_i.
    # 3) At the break i, we have T_i = T_{i+1}. Call that bit b. Because T_i was determined by
    #    the prefix pattern p, we have b = T_i = altBit(p, i-1). 
    #    Then T_{i+1} is forced to be the same b. We may have to pay a flip cost for S_{i+1}
    #    if S_{i+1} != b.
    # 4) From position i+2 onward, T must alternate again, but now T_{i+1} = b, so T_{i+2} = 1-b,
    #    T_{i+3} = b, etc., up to T_N.
    #
    # Thus the cost to construct T under the choice (i, p) is:
    #    cost of matching S_1..S_i to an alternating pattern starting with p
    #    + (possible flip cost at i+1 if S_{i+1} != b)
    #    + cost of matching S_{i+2}..S_N to an alternating pattern that starts with (1-b) at index i+2.
    #
    # We precompute prefix costs for two alternating patterns (start with 0 or start with 1),
    # and suffix costs for two alternating patterns (start with 0 or start with 1) at each index.
    #
    # Let costPref0[i] = minimal cost to match S[1..i] to the alternating pattern T_1=0, T_2=1, ...
    # Let costPref1[i] = minimal cost to match S[1..i] to the alternating pattern T_1=1, T_2=0, ...
    #
    # Let costSuf0[i] = minimal cost to match S[i..N] to an alternating pattern that starts
    #    with T_i=0 (then T_{i+1}=1, T_{i+2}=0, etc.)
    # Let costSuf1[i] = minimal cost to match S[i..N] to an alternating pattern that starts
    #    with T_i=1 (then T_{i+1}=0, T_{i+2}=1, etc.)
    #
    # Compute these in O(N).  Then for each i in [1..N-1] and each p in {0,1}:
    #   b = altBit(p, i-1) is T_i (the i-th bit of the prefix pattern p).
    #   flipCost = cost if we must flip S_{i+1} to b (if S_{i+1} != b).
    #   suffixCost = costSuf(1-b)[ i+2 ], because from i+2 onward we want T_{i+2} = 1-b, T_{i+3}=b, ...
    #   total = costPrefp[i] + flipCost + suffixCost
    # Take the minimum over all choices.
    #
    # Edge cases: 
    #  - If i+2 > N, that suffix cost is zero by definition (no further positions).
    #  - Make sure indexing carefully matches up.
    #
    # Time complexity: O(N).

    # ------------------------------------------------------------
    # Helper for mismatch cost: returns cost if we want to force bit b on S[i].
    # b and S[i] are in {0,1}. C[i] is the cost to flip from S[i] to 1-S[i].
    # mismatch_cost(i, b) = C[i] if S[i] != b else 0
    def mismatch_cost(idx, b):
        return 0 if S[idx] == b else C[idx]

    # ------------------------------------------------------------
    # Build prefix arrays costPref0 and costPref1
    # costPref0[i] = cost to match S[0..i-1] with (0,1,0,1,...) for T[1..i]
    # That is, T_j = altBit(0, j-1) for j in [1..i].
    # We'll do 1-based for i in code but remember S/C are 0-based.
    costPref0 = [0]*(N+1)
    costPref1 = [0]*(N+1)

    # altBit(p, x) = p ^ (x % 2).  If x is even => same as p, if x is odd => 1-p.
    # For T_j with start p, T_j = altBit(p, j-1).

    for i in range(1, N+1):
        # previous cost
        costPref0[i] = costPref0[i-1] + (C[i-1] if S[i-1] != ((0) ^ ((i-1) & 1)) else 0)
        costPref1[i] = costPref1[i-1] + (C[i-1] if S[i-1] != ((1) ^ ((i-1) & 1)) else 0)

    # ------------------------------------------------------------
    # Build suffix arrays costSuf0 and costSuf1
    # costSuf0[i] = cost to match S[i-1..N-1] to pattern T_i=0, T_{i+1}=1, ...
    # i in [1..N], in code we store costSuf0[i], costSuf1[i], plus define costSuf0[N+1], costSuf1[N+1] = 0
    # We'll fill i in descending order.  Relationship:
    #   costSuf0[i] = mismatch_cost(i-1, 0) + costSuf1[i+1]  (since next bit must be 1)
    #   costSuf1[i] = mismatch_cost(i-1, 1) + costSuf0[i+1]  (since next bit must be 0)
    #
    # We'll define costSuf0[N+1] = costSuf1[N+1] = 0 as a base case.

    costSuf0 = [0]*(N+2)
    costSuf1 = [0]*(N+2)

    # Fill from i=N down to i=1
    for i in range(N, 0, -1):
        costSuf0[i] = mismatch_cost(i-1, 0) + costSuf1[i+1]
        costSuf1[i] = mismatch_cost(i-1, 1) + costSuf0[i+1]

    # ------------------------------------------------------------
    # Now iterate over i in [1..N-1], and pattern start p in {0,1}.
    #  - The prefix cost = costPrefp[i].
    #  - T_i = altBit(p, i-1).
    #  - Must set T_{i+1} = T_i => flip cost for S_{i+1} if needed.
    #  - Then from i+2 onward, T_{i+2} = 1 - T_{i+1}, so suffix cost = costSuf(1-b)[i+2],
    #    where b is T_i.
    #
    # We'll track the minimum.

    INF = 10**20
    ans = INF

    # altBit(p, x) = p ^ (x%2)
    def altBit(p, x):
        return p ^ (x & 1)

    for i in range(1, N):  # i in [1..N-1]
        for p in (0, 1):
            # The prefix up to i uses pattern p. That cost:
            prefix_cost = costPref0[i] if p == 0 else costPref1[i]
            # b = T_i
            b = altBit(p, i-1)
            # cost to flip S_{i+1} to b (S index i -> i+1 in 1-based)
            flip_cost = C[i] if S[i] != b else 0

            # suffix cost: from i+2 onward, the bit at i+2 must be (1-b).
            # i+2 in 1-based => index i+1 in 0-based
            # costSuf(1-b)[ i+2 ] if i+2 <= N, else 0
            if i+2 <= N:
                suffix_cost = costSuf0[i+2] if (1-b) == 0 else costSuf1[i+2]
            else:
                suffix_cost = 0

            total = prefix_cost + flip_cost + suffix_cost
            if total < ans:
                ans = total

    print(ans)