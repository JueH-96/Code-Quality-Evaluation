def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    S = next(it).strip()
    C = [int(next(it)) for _ in range(N)]
    
    # We want to transform S into a "good string" T where exactly one adjacent pair (i and i+1)
    # are equal. It turns out that any good string T must be exactly alternating everywhere except
    # at one “break” between positions i and i+1 (for some i in 1..N-1) where T_i = T_{i+1}.
    #
    # For a fixed break position i and a chosen bit X for the break pair,
    # we must have:
    #    For j = 1..i:
    #        T_j = X           if (i - j) is even,
    #        T_j = 1 - X       if (i - j) is odd.
    #    For j = i+1..N:
    #        T_j = X           if (j - i - 1) is even,
    #        T_j = 1 - X       if (j - i - 1) is odd.
    #
    # We can rephrase these formulas in a uniform way in terms of j's index parity:
    #   For j from 1 to i, note that (i - j) mod 2 equals (i mod 2) XOR (j mod 2).
    #   Hence, for j <= i, we have:
    #       T_j = (X XOR (i mod 2)) XOR (j mod 2)
    #   Similarly, for j >= i+1, since (j - i - 1) mod 2 is (j mod 2) XOR ((i+1) mod 2),
    #       T_j = (X XOR ((i+1) mod 2)) XOR (j mod 2)
    #
    # So if we define:
    #   L = X XOR (i mod 2)  and  R = X XOR ((i+1) mod 2),
    # then the desired bit at position j is:
    #   - For j ≤ i: desired bit = (L XOR (j mod 2))
    #   - For j ≥ i+1: desired bit = (R XOR (j mod 2))
    #
    # Notice that these two formulas are exactly “alternating” patterns but with a fixed offset.
    # We define two canonical alternating patterns over positions 1..N:
    #   Pattern P0: at position j (1-indexed), target bit = (j mod 2)
    #               (i.e. odd positions 1, even positions 0)
    #   Pattern P1: at position j, target bit = 1 - (j mod 2)
    #
    # Thus if at a segment we want the pattern "b XOR (j mod 2)" where b is 0,
    # that means we want P0; if b = 1, we want P1.
    #
    # We can precompute for each position j the cost of converting S[j] to
    # the bit required by P0 and P1. Note that positions here are 1-indexed.
    # For position j (1-indexed), let:
    #   cost0[j] = cost to make S[j] become (j mod 2)   [i.e., P0],
    #   cost1[j] = cost to make S[j] become (1 - (j mod 2)) [i.e., P1].
    
    # Build the cost arrays (using 0-indexed array positions corresponding to j=1..N)
    cost0 = [0] * N
    cost1 = [0] * N
    for j in range(N):
        pos = j + 1  # convert to 1-indexed
        # For P0, desired is:
        #   if pos is odd, desired is '1'; if even, desired is '0'
        if pos % 2 == 1:
            desired = '1'
        else:
            desired = '0'
        if S[j] == desired:
            cost0[j] = 0
        else:
            cost0[j] = C[j]
        # For P1, desired is the opposite: if pos is odd, desired is '0'; if even, desired is '1'
        desired_inv = '0' if desired == '1' else '1'
        if S[j] == desired_inv:
            cost1[j] = 0
        else:
            cost1[j] = C[j]
    
    # Precompute prefix sums for cost0 and cost1.
    # Let prefix0[i] be the total cost for positions 1..i to match P0.
    # Similarly for prefix1.
    prefix0 = [0] * (N + 1)
    prefix1 = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix0[i] = prefix0[i - 1] + cost0[i - 1]
        prefix1[i] = prefix1[i - 1] + cost1[i - 1]
    
    # Now, for each possible break position i (1 ≤ i ≤ N-1), we try both choices for X in {0,1}.
    #
    # For j ≤ i:
    #   The desired pattern is (X XOR (i mod 2)) XOR (j mod 2).
    #   If we let L = X XOR (i mod 2):
    #     if L == 0, then the segment should match P0, costing prefix0[i],
    #     if L == 1, then it should match P1, costing prefix1[i].
    #
    # For j ≥ i+1:
    #   The desired pattern is (X XOR ((i+1) mod 2)) XOR (j mod 2).
    #   Let R = X XOR ((i+1) mod 2):
    #     if R == 0, then cost = prefix0[N] - prefix0[i],
    #     if R == 1, then cost = prefix1[N] - prefix1[i].
    
    # We iterate over i from 1 to N-1 (using 1-indexing) and try both X=0 and X=1.
    ans = 10**20
    for i in range(1, N):
        # a = i mod 2 and b = (i+1) mod 2 (remember: i is 1-indexed)
        a = i % 2
        b = (i + 1) % 2
        
        # Candidate with X = 0:
        #   L = 0 XOR a = a and R = 0 XOR b = b.
        if a == 0:
            left_cost = prefix0[i]
        else:
            left_cost = prefix1[i]
        if b == 0:
            right_cost = prefix0[N] - prefix0[i]
        else:
            right_cost = prefix1[N] - prefix1[i]
        cost_candidate0 = left_cost + right_cost
        
        # Candidate with X = 1:
        #   L = 1 XOR a and R = 1 XOR b.
        #   Note: 1 XOR a is 0 when a == 1, and 1 when a == 0.
        if a == 1:
            left_cost = prefix0[i]
        else:
            left_cost = prefix1[i]
        if b == 1:
            right_cost = prefix0[N] - prefix0[i]
        else:
            right_cost = prefix1[N] - prefix1[i]
        cost_candidate1 = left_cost + right_cost
        
        candidate_cost = min(cost_candidate0, cost_candidate1)
        if candidate_cost < ans:
            ans = candidate_cost
    
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()