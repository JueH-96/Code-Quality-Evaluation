def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast I/O pointers
    # We'll parse as we go.
    # N, Q
    N = int(input_data[0])
    Q = int(input_data[1])
    S_str = input_data[2]
    
    # Convert S to 1-based list of integer bits for convenience
    S = [0]*(N+1)
    for i in range(N):
        S[i+1] = 1 if S_str[i] == '1' else 0
    
    # We will maintain:
    # 1) A Fenwick (Binary Indexed) Tree "flipFT" to record flip operations in a difference-array manner.
    #    So flipFT[i] tracks partial sums that let us compute how many flips affect position i (mod 2).
    # 2) A Fenwick Tree "badFT" to track how many adjacent pairs are "bad" (i.e. the same).
    #    bad[i] = 1 if (final S[i] == final S[i+1]) else 0, for i in [1..N-1].
    #
    # For a query of type 2 (L, R), the substring is good iff the sum of "bad" in [L..R-1] is 0.
    #
    # When flipping [L,R], only the adjacency between (L-1,L) and (R,R+1) can change
    # (provided L>1 or R<N, respectively), because flipping a contiguous block does not change
    # whether two positions strictly inside the block match or differ.
    #
    # We'll implement 1-based Fenwicks for both structures.

    # Fenwicks typically: fenwicks[i] = sum over range [ i - (i & -i) + 1 .. i ]
    # We'll define functions:
    #   fenwicks_update(BIT, pos, val) -> adds val to index pos
    #   fenwicks_sum(BIT, pos) -> returns sum from 1..pos
    # We'll do everything mod 2 for flipFT, but we only need sums mod 2 from it,
    # so we can store normal sums and just take sum(...) % 2 when we getFlip(pos).
    # For badFT we store normal integer counts.

    sys.setrecursionlimit(10**7)

    # Fenwicks for flip array
    flipFT = [0]*(N+2)  # We'll use indices up to N+1 possibly.

    def flip_update(idx, val):
        """ Add 'val' to index 'idx' in flip Fenwicks (1-based). """
        while idx <= N:
            flipFT[idx] += val
            idx += idx & -idx

    def flip_sum(idx):
        """ Returns sum(1..idx) in flip Fenwicks (1-based). """
        s = 0
        while idx > 0:
            s += flipFT[idx]
            idx -= idx & -idx
        return s

    def getFlip(pos):
        """ Returns whether position 'pos' is flipped (mod 2). """
        return flip_sum(pos) & 1

    # Build the "bad" array for the initial S (without flips).
    bad = [0]*(N)  # bad[i] for i in [1..N-1], but we'll store in 1-based: bad[i] means adjacency i-(i+1).
    for i in range(1, N):
        bad[i] = 1 if S[i] == S[i+1] else 0

    # Fenwicks for bad counts
    badFT = [0]*(N+1)   # We'll use size N, storing bad[i] in i.

    def bad_update(idx, diff):
        """Add 'diff' to badFT at position idx (1-based)."""
        while idx <= N-1:
            badFT[idx] += diff
            idx += idx & -idx

    def bad_sum(idx):
        """Returns sum of bad[1..idx] in 1-based indexing."""
        s = 0
        while idx > 0:
            s += badFT[idx]
            idx -= idx & -idx
        return s

    # Build the Fenwicks tree for bad initially.
    for i in range(1, N):
        # we add bad[i] to position i
        val = bad[i]
        j = i
        while j <= N-1:
            badFT[j] += val
            j += j & -j

    # A helper to recalc adjacency at index i (meaning the pair (i, i+1)), then
    # update the Fenwicks tree for that position. i in [1..N-1].
    def recalc_bad(i):
        if 1 <= i < N:
            # old value
            old = bad[i]
            # compute new final bits for S[i], S[i+1]
            si = S[i] ^ getFlip(i)
            si1 = S[i+1] ^ getFlip(i+1)
            new = 1 if si == si1 else 0
            bad[i] = new
            if new != old:
                bad_update(i, new - old)

    # Process queries
    # The queries start from input_data[3] onward
    ptr = 3
    out = []
    for _ in range(Q):
        t = int(input_data[ptr]); ptr+=1
        L = int(input_data[ptr]); ptr+=1
        R = int(input_data[ptr]); ptr+=1

        if t == 1:
            # flip L..R
            # 1) update flip Fenwicks
            flip_update(L, 1)
            if R+1 <= N:
                flip_update(R+1, -1)
            # 2) recalc adjacency for (L-1) if valid
            if L > 1:
                recalc_bad(L-1)
            # 3) recalc adjacency for R if valid
            if R < N:
                recalc_bad(R)
        else:
            # t == 2, query L..R
            if L == R:
                # single character is always good
                out.append("Yes")
            else:
                # sum of bad in [L..R-1]
                s_val = bad_sum(R-1) - bad_sum(L-1)
                if s_val == 0:
                    out.append("Yes")
                else:
                    out.append("No")

    print("
".join(out))

# Don't forget to call main()
if __name__ == "__main__":
    main()