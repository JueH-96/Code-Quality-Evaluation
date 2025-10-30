def main():
    import sys
    data = sys.stdin
    # Read N (board size) and M (number of existing pieces)
    line = data.readline().split()
    N = int(line[0])
    M = int(line[1])
    # We'll encode each occupied square (i,j) as a single integer key = (i<<32)|j
    # This lets us store positions in a single set of ints.
    SHIFT = 32
    pos_set = set()
    for _ in range(M):
        a_str, b_str = data.readline().split()
        a = int(a_str)
        b = int(b_str)
        pos_set.add((a << SHIFT) | b)

    # The 8 knight‐moves
    moves = (
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1),
    )

    attacked_set = set()
    # A mask to recover the low 32 bits for the column index
    mask_low = (1 << SHIFT) - 1

    # For each occupied piece, mark its attacked squares
    for key in pos_set:
        # decode row (a) and column (b)
        a = key >> SHIFT
        b = key & mask_low
        for dx, dy in moves:
            ni = a + dx
            nj = b + dy
            # Check bounds
            if not (1 <= ni <= N and 1 <= nj <= N):
                continue
            new_key = (ni << SHIFT) | nj
            # Skip if already occupied or already counted as attacked
            if new_key in pos_set or new_key in attacked_set:
                continue
            attacked_set.add(new_key)

    attacked_count = len(attacked_set)
    # Total safe squares = N^2 (all) - M (occupied) - attacked_count
    # Python ints handle big values fine.
    total_safe = N * N - M - attacked_count
    sys.stdout.write(str(total_safe))


if __name__ == "__main__":
    main()