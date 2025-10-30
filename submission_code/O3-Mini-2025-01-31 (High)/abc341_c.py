def main():
    import sys
    import numpy as np
    data = sys.stdin.read().splitlines()
    if not data:
        return

    # Parse input: H rows, W columns, and N moves.
    H, W, N = map(int, data[0].split())
    T = data[1].strip()
    grid_lines = data[2:2+H]

    # Create a boolean grid "land" where True means cell is land ('.'),
    # and False means sea ('#').
    land = np.array([[ch == '.' for ch in line] for line in grid_lines], dtype=bool)

    # Compute the relative offsets from the starting cell.
    # offsets[k] is the displacement after k moves.
    offsets = [(0, 0)]
    dr = 0
    dc = 0
    for move in T:
        if move == 'L':
            dc -= 1
        elif move == 'R':
            dc += 1
        elif move == 'U':
            dr -= 1
        elif move == 'D':
            dr += 1
        offsets.append((dr, dc))

    # For a starting position (i, j) to be valid,
    # for all k (0 <= k <= N) the cell (i + dr[k], j + dc[k]) must be land.
    # Instead of checking every candidate cell repeatedly, we take the set
    # of unique offsets (displacements) visited along the path.
    unique_offsets = set(offsets)
    # Note: the net offset (last offset) is used to compute the final position,
    # but since translation is one-to-one, counting the valid starting cells
    # is equivalent to counting the distinct final positions.

    # valid will mark candidate starting cells which satisfy the condition for all offsets.
    valid = np.ones((H, W), dtype=bool)

    # For each offset in the unique set, update the valid starting positions.
    for r_off, c_off in unique_offsets:
        # For a candidate starting cell (i,j), the visited cell is (i+r_off, j+c_off).
        # We need (i+r_off, j+c_off) to be within the grid.
        # Determine the allowable index range for i and j.
        row_low = max(0, -r_off)
        row_high = min(H, H - r_off)
        col_low = max(0, -c_off)
        col_high = min(W, W - c_off)

        # Create a mask for positions where the condition holds for this offset.
        mask = np.zeros((H, W), dtype=bool)
        # For positions in the valid region, check if shifting by (r_off, c_off) lands on land.
        mask[row_low:row_high, col_low:col_high] = land[row_low + r_off : row_high + r_off, col_low + c_off : col_high + c_off]

        # Intersect with the current valid positions.
        valid &= mask

    # The final answer is the count of valid starting positions.
    # Since the current (final) position is the starting cell plus a fixed net offset,
    # this count equals the number of possible current positions.
    result = int(np.count_nonzero(valid))
    print(result)

if __name__ == '__main__':
    main()