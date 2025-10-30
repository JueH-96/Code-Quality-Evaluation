def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    S = data[1]
    C = list(map(int, data[2:]))

    # We are going to build our final string T in two parts. 
    # In our “good” T, there is exactly one pair of consecutive equal bits.
    # If we consider 0-indexing, we choose an index j (0 ≤ j ≤ N–2) so that
    # T[j] = T[j+1] (the unique violation) while every other adjacent pair is alternating.
    # Moreover, T[0..j] will be an alternating sequence (we have two choices for the first element),
    # and T[j+1..N–1] will be forced to alternate starting from T[j].
    #
    # There are exactly 2 choices for the first segment’s pattern:
    #   • Option "b = 0": Use pattern_A where pattern_A[i] = 0 if i even, 1 if odd.
    #   • Option "b = 1": Use pattern_B where pattern_B[i] = 1 if i even, 0 if odd.
    #
    # Once the first segment is fixed, let v be the last bit T[j].
    # Then the second segment (indices j+1 to N–1) must be an alternating sequence
    # starting with v. But note that any alternating sequence on the whole indices is either the global pattern_A or pattern_B.
    # In fact, if we denote:
    #     pattern_A[i] = 0 if i even, 1 if i odd
    #     pattern_B[i] = 1 if i even, 0 if i odd,
    # then for the second segment at index j+1 we can choose:
    #     if v == pattern_A[j+1] then use pattern_A for the rest,
    #     else use pattern_B.
    #
    # We can compute the cost very efficiently if we precompute prefix sums for the cost of converting S into these global patterns.
    #
    # For index i (0-indexed), the cost to change S[i] to match:
    #   pattern_A[i] = (0 if i even, 1 if i odd) 
    #   pattern_B[i] = (1 if i even, 0 if i odd)
    # is C[i] (if S[i] is not equal to the expected bit) or 0.
    #
    # Then for any interval [L, R] (0-indexed) the cost to force to a global pattern is the appropriate difference of prefix sums.
    
    # Precompute prefix sums for the two patterns.
    prefixA = [0] * (N + 1)
    prefixB = [0] * (N + 1)
    for i in range(N):
        expA = 0 if (i % 2 == 0) else 1
        expB = 1 if (i % 2 == 0) else 0
        costA = C[i] if (int(S[i]) != expA) else 0
        costB = C[i] if (int(S[i]) != expB) else 0
        prefixA[i+1] = prefixA[i] + costA
        prefixB[i+1] = prefixB[i] + costB

    ans = float('inf')
    # j will be the last position of the first segment.
    # Then the unique violation happens between indices j and j+1.
    for j in range(0, N - 1):
        # Case 1: use b = 0 for the first segment => use pattern_A on indices 0..j.
        first_seg_cost = prefixA[j+1]
        # In pattern_A, T[j] = 0 if j is even; 1 if odd.
        v = 0 if (j % 2 == 0) else 1

        # For the second segment (indices j+1...N-1) we must have T[j+1] = v.
        # The global patterns at index j+1 are:
        #   pattern_A[j+1] = (0 if (j+1) is even, else 1)
        #   pattern_B[j+1] = (1 if (j+1) is even, else 0)
        expA_second = 0 if ((j + 1) % 2 == 0) else 1
        if v == expA_second:
            second_seg_cost = prefixA[N] - prefixA[j+1]
        else:
            second_seg_cost = prefixB[N] - prefixB[j+1]
        candidate = first_seg_cost + second_seg_cost
        if candidate < ans:
            ans = candidate

        # Case 2: use b = 1 for the first segment => use pattern_B on indices 0..j.
        first_seg_cost = prefixB[j+1]
        # In pattern_B, T[j] = 1 if j is even; 0 if odd.
        v = 1 if (j % 2 == 0) else 0
        expA_second = 0 if ((j + 1) % 2 == 0) else 1
        if v == expA_second:
            second_seg_cost = prefixA[N] - prefixA[j+1]
        else:
            second_seg_cost = prefixB[N] - prefixB[j+1]
        candidate = first_seg_cost + second_seg_cost
        if candidate < ans:
            ans = candidate

    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()