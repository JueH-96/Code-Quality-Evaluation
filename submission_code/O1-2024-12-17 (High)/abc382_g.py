def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    T = int(input_data[0])
    ptr = 1

    # We want to compute floor((x + 0.5) / K) using only integer arithmetic (so as not to lose precision).
    # Note that floor((x + 0.5)/K) = floor((2x + 1) / (2K)).
    # Python's // operator does true floor-division for negatives as well, so this works correctly.
    def tile_index(x, K):
        return (2*x + 1) // (2*K)

    # For a "horizontal" tile (parity of i+j is even), the subcell index is in the y-direction:
    #   subcell = floor( (y + 0.5) - j*K )
    # We can rewrite that safely in integer arithmetic as:
    #   subcell = floor((2y + 1)/2) - j*K
    def horizontal_subcell(y, j, K):
        return ((2*y + 1) // 2) - j*K

    # For a "vertical" tile (parity of i+j is odd), the subcell index is in the x-direction:
    #   subcell = floor( (x + 0.5) - i*K )
    # Same pattern: subcell = floor((2x + 1)/2) - i*K
    def vertical_subcell(x, i, K):
        return ((2*x + 1) // 2) - i*K

    answers = []
    for _ in range(T):
        K   = int(input_data[ptr]); ptr += 1
        Sx  = int(input_data[ptr]); ptr += 1
        Sy  = int(input_data[ptr]); ptr += 1
        Tx  = int(input_data[ptr]); ptr += 1
        Ty  = int(input_data[ptr]); ptr += 1

        # First find which integer "super-tile" (i,j) each point lies in.
        # i = floor((x + 0.5)/K), j = floor((y + 0.5)/K).
        iS = tile_index(Sx, K)
        jS = tile_index(Sy, K)
        iT = tile_index(Tx, K)
        jT = tile_index(Ty, K)

        # If both points happen to lie in the exact same (i, j) super-tile,
        # then the distance is simply how many sub-tile steps we need within that single tile.
        if iS == iT and jS == jT:
            # Orientation is determined by parity of i+j:
            if ((iS + jS) & 1) == 0:
                # Horizontal tile => subcell index depends on y
                s_sub = horizontal_subcell(Sy, jS, K)
                t_sub = horizontal_subcell(Ty, jT, K)
            else:
                # Vertical tile => subcell index depends on x
                s_sub = vertical_subcell(Sx, iS, K)
                t_sub = vertical_subcell(Tx, iT, K)
            answers.append(str(abs(s_sub - t_sub)))
            continue

        # Otherwise, let d = the Manhattan distance in the (i,j)-grid:
        d = abs(iS - iT) + abs(jS - jT)

        # In this tiling problem, one extra step can appear precisely when the parity of
        # (iS+jS) differs from the parity of (iT+jT).  Otherwise, it is just d.
        # (Reason: effectively, one often needs one extra "sub-tile" move if the start
        #  and target tiles belong to different parity-classes in the tiling adjacency.)
        pS = (iS + jS) & 1
        pT = (iT + jT) & 1
        overhead = 1 if pS != pT else 0

        answers.append(str(d + overhead))

    print("
".join(answers))