def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    # N is an even number.
    try:
        N = int(data[0])
    except:
        return
    grid = data[1:1+N]
    # We work in 0-indexed coordinates.
    # A key observation:
    #
    # In the problem, we perform a “rotation‐by‐90 degrees” operation on the subgrid
    # with indices i ... (N+1-i) (in 1-indexed notation) for i = 1, 2, …, N/2.
    #
    # In other words, every cell (x,y) in the subgrid gets “moved” by the rule:
    #   new cell at (y, N+1-x) gets the old color from (x,y).
    #
    # When we interpret this on the whole grid (for i = 1 the entire grid is rotated)
    # and then “re‐rotate” each inner square for i = 2, 3, …, the net effect is that
    # each cell is updated in every operation for which it lies in the subgrid.
    #
    # If we define the "layer" L of a cell (using 1-indexed layers) as
    #   L = min(r, c, N+1-r, N+1-c)
    # (where (r,c) are the row, column in 1–indexed notation),
    # then a cell in layer L is updated L times.
    #
    # In each operation the update “pulls” the color from the pre–rotation cell
    # according to the mapping T: (x, y) → (y, N+1-x) (90° clockwise).
    # Thus, if you want to know the final color in a cell, you can “trace back”
    # the successive moves. Since the updates are simultaneous in each round,
    # the final color in cell (r, c) (with 1-indexing) is the original color
    # at cell T^(-L)(r, c) where T^(-1) is the inverse mapping.
    #
    # In 1–indexed coordinates the mapping T is:
    #    T(x,y) = (y, N+1-x)
    # Its inverse is
    #    T^(-1)(x,y) = (N+1-y, x)
    #
    # Converting to 0-indexed (i.e. row, col in 0..N–1):
    #    T_ccw (90° counterclockwise) is:
    #       (r, c) → (N-1-c, r)
    # and T_ccw’s inverse is the 90° clockwise map.
    # (Here it is natural to note that T^(-1) from the problem equals a 90° counterclockwise rotation.)
    #
    # Thus a cell at 0-index (r, c) is updated L times (with L = min(r+1, c+1, N-r, N-c))
    # and its final color comes from the original cell at
    #    T_ccw^L (r, c),
    # where T_ccw^L means apply the 90° counterclockwise rotation L times.
    #
    # But note that T_ccw is a 90° rotation so T_ccw^4 is the identity.
    # Therefore, if we let k = L mod 4 then the source cell is:
    #
    #   • if k == 0: (r,       c)
    #   • if k == 1: (N-1-c,   r)
    #   • if k == 2: (N-1-r,   N-1-c)
    #   • if k == 3: (c,       N-1-r)
    #
    # We now compute for every cell (r, c) its layer L and then use the appropriate
    # transformation to fetch the final color from the original grid.
    
    # Precompute for each column the “column‐side minimum”
    # For a given column c (0-indexed) the minimum distance from an edge is min(c+1, N-c)
    col_min = [ (c+1 if c+1 < N-c else N-c) for c in range(N) ]
    
    out_lines = [None] * N
    for r in range(N):
        # For row r, min(r+1, N-r) is the minimum distance from the top or bottom edge.
        row_min = r+1 if r+1 < N-r else N-r
        row_chars = [None] * N
        for c in range(N):
            # The layer number (in 1-indexed terms) is:
            #    L = min(r+1, c+1, N-r, N-c)
            # We have precomputed col_min[c] = min(c+1, N-c) so:
            L = row_min if row_min < col_min[c] else col_min[c]
            mod = L % 4  # the net effect is this many 90° counterclockwise rotations.
            if mod == 0:
                sr, sc = r, c
            elif mod == 1:
                sr, sc = N - 1 - c, r
            elif mod == 2:
                sr, sc = N - 1 - r, N - 1 - c
            else:  # mod == 3
                sr, sc = c, N - 1 - r
            row_chars[c] = grid[sr][sc]
        out_lines[r] = "".join(row_chars)
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()