def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    # Read A
    H_A, W_A = map(int, input().split())
    A = [input().rstrip() for _ in range(H_A)]
    # Read B
    H_B, W_B = map(int, input().split())
    B = [input().rstrip() for _ in range(H_B)]
    # Read X
    H_X, W_X = map(int, input().split())
    X = [input().rstrip() for _ in range(H_X)]

    # Collect black cell coordinates for A and B
    A_cells = [(i, j) for i in range(H_A) for j in range(W_A) if A[i][j] == '#']
    B_cells = [(i, j) for i in range(H_B) for j in range(W_B) if B[i][j] == '#']

    # Compute bounding boxes for A and B blacks
    min_i_A = min(i for i, _ in A_cells)
    max_i_A = max(i for i, _ in A_cells)
    min_j_A = min(j for _, j in A_cells)
    max_j_A = max(j for _, j in A_cells)

    min_i_B = min(i for i, _ in B_cells)
    max_i_B = max(i for i, _ in B_cells)
    min_j_B = min(j for _, j in B_cells)
    max_j_B = max(j for _, j in B_cells)

    # Determine valid translation ranges for A so that all its blacks fall inside the cut of size H_X x W_X
    # If A cell (i,j) maps to (i+dyA, j+dxA), we need 0 <= i+dyA < H_X, 0 <= j+dxA < W_X
    # For all i in [min_i_A, max_i_A], that means
    #   -min_i_A <= dyA <= (H_X-1) - max_i_A
    # Similarly for dxA.
    dyA_min = -min_i_A
    dyA_max = (H_X - 1) - max_i_A
    dxA_min = -min_j_A
    dxA_max = (W_X - 1) - max_j_A

    # And for B
    dyB_min = -min_i_B
    dyB_max = (H_X - 1) - max_i_B
    dxB_min = -min_j_B
    dxB_max = (W_X - 1) - max_j_B

    # If any range is empty, we can't place that sheet at all, so no solution
    if dyA_min > dyA_max or dxA_min > dxA_max or dyB_min > dyB_max or dxB_min > dxB_max:
        print("No")
        return

    # Try all combinations of valid translations
    for dyA in range(dyA_min, dyA_max + 1):
        for dxA in range(dxA_min, dxA_max + 1):
            # Create an empty cut-out area with False=transparent
            # We'll reuse this grid for each B-position attempt, so make a fresh copy in the B-loop
            for dyB in range(dyB_min, dyB_max + 1):
                for dxB in range(dxB_min, dxB_max + 1):
                    # Build the combined result in the H_X x W_X window
                    # False = '.', True = '#'
                    grid = [[False]*W_X for _ in range(H_X)]
                    # Paste A
                    for i, j in A_cells:
                        y = i + dyA
                        x = j + dxA
                        grid[y][x] = True
                    # Paste B
                    for i, j in B_cells:
                        y = i + dyB
                        x = j + dxB
                        grid[y][x] = True

                    # Compare with X
                    ok = True
                    for y in range(H_X):
                        row = grid[y]
                        target = X[y]
                        for x in range(W_X):
                            # If grid says black but X has '.', fail
                            if row[x] and target[x] == '.':
                                ok = False
                                break
                            # If grid says transparent but X has '#', fail
                            if not row[x] and target[x] == '#':
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        print("Yes")
                        return

    # If no placement works
    print("No")

if __name__ == "__main__":
    main()