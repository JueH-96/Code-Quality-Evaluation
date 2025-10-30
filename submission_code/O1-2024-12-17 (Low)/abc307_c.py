def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    ptr = 0

    # Read sheet A
    H_A = int(input_data[ptr]); ptr+=1
    W_A = int(input_data[ptr]); ptr+=1
    A = input_data[ptr: ptr+H_A]
    ptr += H_A

    # Read sheet B
    H_B = int(input_data[ptr]); ptr+=1
    W_B = int(input_data[ptr]); ptr+=1
    B = input_data[ptr: ptr+H_B]
    ptr += H_B

    # Read sheet X
    H_X = int(input_data[ptr]); ptr+=1
    W_X = int(input_data[ptr]); ptr+=1
    X = input_data[ptr: ptr+H_X]
    ptr += H_X

    # Collect black squares (r,c) for each sheet, 0-based
    black_A = []
    for r in range(H_A):
        for c in range(W_A):
            if A[r][c] == '#':
                black_A.append((r, c))

    black_B = []
    for r in range(H_B):
        for c in range(W_B):
            if B[r][c] == '#':
                black_B.append((r, c))

    black_X = set()
    for r in range(H_X):
        for c in range(W_X):
            if X[r][c] == '#':
                black_X.add((r, c))

    # Precompute bounding row/col of black squares for A and B
    # We'll use these to limit possible shifts so that all black squares fit in X's HxW
    if black_A:
        minA_r = min(r for r, _ in black_A)
        maxA_r = max(r for r, _ in black_A)
        minA_c = min(c for _, c in black_A)
        maxA_c = max(c for _, c in black_A)
    else:
        # not expected because problem states each sheet has at least one black square
        # but in principle handle it
        minA_r = maxA_r = minA_c = maxA_c = 0

    if black_B:
        minB_r = min(r for r, _ in black_B)
        maxB_r = max(r for r, _ in black_B)
        minB_c = min(c for _, c in black_B)
        maxB_c = max(c for _, c in black_B)
    else:
        minB_r = maxB_r = minB_c = maxB_c = 0

    # For A, valid shiftAx means all black squares (r+shiftAx) in [0, H_X-1]
    # => shiftAx in [ -r, H_X-1 - r ] for each black square r
    # so overall shiftAx in [  -maxA_r, H_X-1 - minA_r ]? That alone doesn't ensure
    # we keep all squares in range. We need the intersection of constraints for all r.
    # The simplest is:
    #   shiftAx_min = max( -r for (r, _) in black_A )
    #   shiftAx_max = min( (H_X-1 - r) for (r, _) in black_A )
    # but we'll do it by bounding minR, maxR instead:
    #   0 <= minA_r + shiftAx and maxA_r + shiftAx < H_X
    # => -minA_r <= shiftAx <= H_X-1 - maxA_r
    shiftAx_min = -minA_r
    shiftAx_max = H_X - 1 - maxA_r
    shiftAy_min = -minA_c
    shiftAy_max = W_X - 1 - maxA_c

    shiftBx_min = -minB_r
    shiftBx_max = H_X - 1 - maxB_r
    shiftBy_min = -minB_c
    shiftBy_max = W_X - 1 - maxB_c

    # Try all valid shifts for A and B
    for shiftAx in range(shiftAx_min, shiftAx_max+1):
        for shiftAy in range(shiftAy_min, shiftAy_max+1):
            # After shifting A, all black squares must still be inside X's range
            # We'll quickly check
            # though we derived shift range for that, let's be consistent
            validA = True
            for (rA, cA) in black_A:
                rr = rA + shiftAx
                cc = cA + shiftAy
                if not (0 <= rr < H_X and 0 <= cc < W_X):
                    validA = False
                    break
            if not validA:
                continue

            for shiftBx in range(shiftBx_min, shiftBx_max+1):
                for shiftBy in range(shiftBy_min, shiftBy_max+1):
                    # Check if all black squares of B remain inside
                    validB = True
                    for (rB, cB) in black_B:
                        rr = rB + shiftBx
                        cc = cB + shiftBy
                        if not (0 <= rr < H_X and 0 <= cc < W_X):
                            validB = False
                            break
                    if not validB:
                        continue

                    # Now form the union of black squares from A and B in X-space
                    unionAB = set()
                    for (rA, cA) in black_A:
                        unionAB.add((rA + shiftAx, cA + shiftAy))
                    for (rB, cB) in black_B:
                        unionAB.add((rB + shiftBx, cB + shiftBy))

                    # Check exact match with X's black squares:
                    # 1) unionAB must be subset of black_X
                    # 2) black_X must be subset of unionAB (just for the region of X)
                    # Actually we want equality: unionAB == black_X
                    if unionAB == black_X:
                        print("Yes")
                        return

    # If no arrangement worked
    print("No")

# Call main
if __name__ == "__main__":
    main()