def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    # Read input
    N = int(data[0])
    grid = data[1:]
    # We are given an N×N grid (with even N) and we must perform a series of operations.
    # For i = 1, 2, …, N/2 (1-indexed) we do:
    #   For every cell (x,y) with x,y in [i, N+1-i] we replace the color in cell (y, N+1-x)
    #   with the color of (x,y) (all simultaneously).
    # Converting to 0-index:
    #   For iteration i from 0 to N/2-1, for every cell (r, c) with r,c in [i, N-i-1],
    #   we set B[r'][c'] = A[r][c] where
    #        (r', c') = ( (y-1), (N+1 - x - 1) ) = ( c, N - 1 - r ).
    #
    # Notice that in every iteration the mapping is the same function f:
    #    f(r,c) = (c, N-1-r)
    #
    # However, the operation is applied sequentially on nested subgrids.
    # A key observation (which one may check with small cases) is that every cell is “rotated”
    # as many times as the number of iterations (levels) in which it lies.
    #
    # In fact, if you define the “layer” of a cell (r, c) (0-indexed) as:
    #     L = min(r, c, N-1-r, N-1-c)
    # then cell (r, c) is in the subgrid for iterations i = 0,1,...,L.
    # (Because in iteration i the cells affected must satisfy i <= r,c <= N-i-1.)
    # So the total number of rotations applied (each 90° clockwise) is (L + 1).
    # However, note that our rotation function f satisfies f⁴ = identity.
    # Thus the effective rotation is ((L+1) mod 4).
    #
    # We recall:
    #   f^0(r,c) = (r, c)
    #   f^1(r,c) = (c, N-1-r)
    #   f^2(r,c) = (N-1-r, N-1-c)
    #   f^3(r,c) = (N-1-c, r)
    #
    # Thus, for each cell (r,c) we compute:
    #    layer = min(r, c, N-1-r, N-1-c)
    #    rot = (layer + 1) % 4   (number of 90° cw rotations)
    # and then set its final location accordingly.
    
    # Prepare an empty answer grid (mutable list of lists)
    ans = [[''] * N for _ in range(N)]
    for r in range(N):
        row = grid[r]
        for c in range(N):
            # Compute layer of (r, c)
            # The layer is the minimum distance from (r,c) to any side.
            layer = r
            if c < layer:
                layer = c
            if N - 1 - r < layer:
                layer = N - 1 - r
            if N - 1 - c < layer:
                layer = N - 1 - c
            # Total rotations: (layer + 1), but rotation is mod 4
            rot = (layer + 1) % 4
            if rot == 0:
                nr, nc = r, c
            elif rot == 1:
                nr, nc = c, N - 1 - r
            elif rot == 2:
                nr, nc = N - 1 - r, N - 1 - c
            else:  # rot == 3
                nr, nc = N - 1 - c, r
            ans[nr][nc] = row[c]
    
    # Output the final grid
    out_lines = ["".join(line) for line in ans]
    sys.stdout.write("
".join(out_lines))
    
if __name__ == "__main__":
    main()