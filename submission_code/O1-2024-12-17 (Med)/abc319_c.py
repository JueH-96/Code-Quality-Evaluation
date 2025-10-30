def main():
    import sys
    sys.setrecursionlimit(10**7)

    # Read the 3x3 grid
    c = []
    for _ in range(3):
        c.extend(list(map(int, sys.stdin.readline().split())))

    # We will index squares in row-major order:
    # 0 -> (0,0), 1 -> (0,1), 2 -> (0,2)
    # 3 -> (1,0), 4 -> (1,1), 5 -> (1,2)
    # 6 -> (2,0), 7 -> (2,1), 8 -> (2,2)
    # c[i] gives the value in that square.

    # Define the 8 lines (rows, columns, diagonals) by their square indices
    lines = [
        (0, 1, 2),  # row 0
        (3, 4, 5),  # row 1
        (6, 7, 8),  # row 2
        (0, 3, 6),  # col 0
        (1, 4, 7),  # col 1
        (2, 5, 8),  # col 2
        (0, 4, 8),  # main diagonal
        (6, 4, 2)   # anti-diagonal
    ]

    # We only care about lines that contain exactly 2 squares of the same value
    # and 1 square of a distinct value, because only such lines can cause
    # the "disappointment" scenario (first two seen are the same, the third is different).
    # If a line has 3 distinct values, it cannot cause disappointment.
    # If a line had 3 identical values, it's forbidden by the problem statement anyway.
    
    # pairSquares[l] = (p1, p2) for the two squares that share the same value
    # distinctSquare[l] = d for the remaining square that has a different value
    # If the line has no duplicates, or is invalid, we store None.
    pairSquares = [None]*8
    distinctSquare = [None]*8

    for l in range(8):
        a, b, d = lines[l]
        va, vb, vd = c[a], c[b], c[d]
        if va == vb and va != vd:
            pairSquares[l] = (a, b)
            distinctSquare[l] = d
        elif va == vd and va != vb:
            pairSquares[l] = (a, d)
            distinctSquare[l] = b
        elif vb == vd and vb != va:
            pairSquares[l] = (b, d)
            distinctSquare[l] = a
        # else: either all three distinct or impossibly all same
        # in both cases, we leave pairSquares[l] = None
    
    # For each square, we collect which lines can cause disappointment
    # (i.e., the lines that have a pairSquares != None and that contain this square).
    lineOfSquare = [[] for _ in range(9)]
    for l in range(8):
        if pairSquares[l] is not None:
            a, b, d = lines[l]
            lineOfSquare[a].append(l)
            lineOfSquare[b].append(l)
            lineOfSquare[d].append(l)

    used = [False]*9  # track which squares are placed in the current permutation
    n_valid_permutations = 0

    # Check if placing newSq causes a guaranteed future disappointment
    # That happens if there's a line l = (p1, p2) pair, distinct d, such that
    # p1 and p2 are already used, and d is not used (different value),
    # meaning eventually we will see that 'd' after the two identical squares => disappointment.
    def causes_disappointment(newSq):
        for l in lineOfSquare[newSq]:
            p12 = pairSquares[l]
            if p12 is None:
                continue
            p1, p2 = p12
            d = distinctSquare[l]
            # If the pair is fully used and the distinct is not used => disappointment
            if used[p1] and used[p2] and not used[d]:
                return True
        return False

    def dfs(depth):
        nonlocal n_valid_permutations
        if depth == 9:
            n_valid_permutations += 1
            return
        for sq in range(9):
            if not used[sq]:
                used[sq] = True
                # Check if this placement triggers a guaranteed disappointment
                if not causes_disappointment(sq):
                    dfs(depth + 1)
                used[sq] = False

    dfs(0)

    from math import factorial
    total_permutations = factorial(9)
    ans = n_valid_permutations / total_permutations

    # Print with sufficient precision
    print(f"{ans:.15f}")

# Do not forget to call main()
if __name__ == "__main__":
    main()