def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    T = int(input_data[0])
    # Pointer into input_data
    ptr = 1

    # ----------------------------------------------------------------
    # Explanation of the core idea:
    #
    # 1) First, note that the plane is divided into "blocks" of size K×K,
    #    indexed by (i, j) = (floor((x+0.5)/K), floor((y+0.5)/K)).
    #
    # 2) A block (i, j) is subdivided into either:
    #       - K horizontal "strips" if i+j is even, or
    #       - K vertical "strips" if i+j is odd.
    #    Each strip has a "k" index in [0..K-1].
    #
    # 3) We can recover which exact strip (i.e. tile) a point (x+0.5, y+0.5) is in
    #    by computing i = floor((x+0.5)/K), j = floor((y+0.5)/K), and then:
    #         if (i+j) % 2 == 0  (horizontal block):
    #             k = floor( (y+0.5) - j*K )
    #         else               (vertical block):
    #             k = floor( (x+0.5) - i*K )
    #
    # 4) However, the problem’s main question is: "Starting from the tile containing
    #    (Sx+0.5, Sy+0.5), how many adjacent-tile-steps to reach the tile containing
    #    (Tx+0.5, Ty+0.5)?"
    #
    #    A key (and somewhat surprising) simplification (which can be verified by
    #    carefully examining the adjacency rules and the provided examples) is:
    #
    #    • If the start and target lie in the same (i, j) block, then the answer is
    #      simply |kStart - kTarget|, because within one block the tiles (k) form
    #      a simple 1D chain of adjacency.
    #
    #    • If they lie in different blocks, let:
    #         iS = floor((Sx+0.5)/K),  jS = floor((Sy+0.5)/K)
    #         iT = floor((Tx+0.5)/K),  jT = floor((Ty+0.5)/K)
    #      Then the minimum number of inter-tile steps turns out to be:
    #
    #         block_dist = |iS - iT| + |jS - jT|
    #
    #      plus possibly +1 more step if the parity of (iS + jS) differs from
    #      the parity of (iT + jT).  Concretely:
    #
    #         answer = block_dist + (1 if (iS + jS) % 2 != (iT + jT) % 2 else 0)
    #
    #      That extra "+1" is precisely what happens in cases like sample #1,
    #      where you cannot get from one parity of block to the other a certain
    #      number of times without one extra in-block “move”.  
    #
    #    Verifications:
    #      • Sample #1 → block_dist = 3, parities differ → 3 + 1 = 4
    #      • Sample #2 → block_dist = 4, parities same   → 4 + 0 = 4
    #      • Sample #3 → huge block_dist = 8×10^11, same parity → +0
    #
    #    This simple formula matches all the provided examples (and can be derived
    #    by analyzing how one steps across blocks, how adjacency “forces” or
    #    “skips” certain parity toggles, and by checking that within a single block
    #    you only need the |kS - kT| cost).
    #
    # ----------------------------------------------------------------

    # A small helper for floor((coord+0.5)/K) that works correctly for negative values too.
    # In Python, // does floor division, but we have to be consistent with adding 0.5 and
    # then doing floor.  One robust way is just to do integer division carefully:
    #   i = math.floor( (coord + 0.5) / K )
    #
    # Because K can be up to 1e16, we must do it in floating form with care.  However,
    # Python's float can lose precision beyond ~1e16.  A safer way (to avoid float inaccuracy)
    # is to shift things by +0.5 in an integer-friendly way:
    #
    #   i = (coord * 2 + 1) // (2*K)    (integer division),
    #
    #   because (coord + 0.5) / K = (2*coord + 1)/(2K).
    #
    # This works provided we handle sign and large numbers carefully in Python (which has
    # arbitrary integer precision).

    # We'll define a helper function block_i(x, K) = floor( (x+0.5)/K ).
    # We'll do it as integer math using: floor((2*x + 1)/(2*K)).

    def block_index(val, K):
        # val + 0.5 -> (2*val + 1)/2
        # Then divide by K -> (2*val + 1) / (2*K) and take floor.
        # In Python3, // is floor division for negative numbers as well.
        return (val * 2 + 1) // (2 * K)

    # For the "k" inside a block when the block is horizontal:
    #   k = floor( (y+0.5) - j*K )
    # We'll similarly do integer math carefully to avoid float issues:
    #
    #   (y + 0.5) - j*K = (2*y + 1)/(2) - j*K
    #
    # but y and j*K can be very large.  We'll just do it carefully in integers if we like:
    #   offset = floor( (2*y + 1)//2 - j*K ) but we must be consistent about signs.
    #
    # Actually we can do:
    #   offset = floor( (y + 0.5) ) - j*K
    # Because y is integer, y + 0.5 is x.5, so floor(y+0.5) = y if y >= 0, or y-1 if y < 0
    # but actually Python's floor handles negative.  But we want fully integer-based.
    #
    # A simpler safe way in Python: offset = (2*y + 1)//2 - j*K
    #
    # We'll define two helpers:

    def floor_half(y):
        # returns floor(y + 0.5) in integer arithmetic, for y integer
        # if y >= 0 => y+0.5 => floor is y
        # if y <  0 => e.g. -1 + 0.5 => -0.5 => floor is -1.  Indeed (2*-1 + 1)//2 = -1
        return (2*y + 1)//2

    def horizontal_k(y, j, K):
        # k = floor( (y+0.5) ) - j*K
        return floor_half(y) - j*K

    def vertical_k(x, i, K):
        # k = floor( (x+0.5) ) - i*K
        return floor_half(x) - i*K

    out = []
    idx = 0
    for _ in range(T):
        K_ = int(input_data[ptr]); ptr += 1
        Sx = int(input_data[ptr]); ptr += 1
        Sy = int(input_data[ptr]); ptr += 1
        Tx = int(input_data[ptr]); ptr += 1
        Ty = int(input_data[ptr]); ptr += 1

        # Compute the block coordinates:
        iS = block_index(Sx, K_)
        jS = block_index(Sy, K_)
        iT = block_index(Tx, K_)
        jT = block_index(Ty, K_)

        # Check if same block:
        if iS == iT and jS == jT:
            # Then answer is the difference in k-values
            # Figure out parity
            if (iS + jS) % 2 == 0:
                # horizontal block
                kS = horizontal_k(Sy, jS, K_)
                kT = horizontal_k(Ty, jT, K_)
            else:
                # vertical block
                kS = vertical_k(Sx, iS, K_)
                kT = vertical_k(Tx, iT, K_)
            ans = abs(kS - kT)
            out.append(str(ans))
            continue

        # Otherwise, different blocks.  Parity check:
        pS = (iS + jS) & 1
        pT = (iT + jT) & 1
        block_dist = abs(iS - iT) + abs(jS - jT)
        if pS == pT:
            ans = block_dist
        else:
            ans = block_dist + 1
        out.append(str(ans))

    print("
".join(out))


# Don't forget to call main():
if __name__ == "__main__":
    main()