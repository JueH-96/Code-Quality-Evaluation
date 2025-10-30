def main():
    import sys

    # Read input
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    pieces = []
    for k in range(M):
        a = int(data[2 + 2*k])
        b = int(data[3 + 2*k])
        pieces.append((a, b))

    # Find forbidden rows and columns
    forbidden_rows = set()
    forbidden_cols = set()
    forbidden_iplusj = set()
    forbidden_iminusj = set()

    for a, b in pieces:
        forbidden_rows.add(a)
        forbidden_cols.add(b)
        forbidden_iplusj.add(a + b)
        forbidden_iminusj.add(a - b)

    R = len(forbidden_rows)
    C = len(forbidden_cols)
    D_sum = len(forbidden_iplusj)
    D_diff = len(forbidden_iminusj)

    # Calculate lengths of forbidden i+j diagonals
    sum_lengths = 0
    for s in forbidden_iplusj:
        if 2 <= s <= N + 1:
            sum_lengths += s - 1
        elif N + 1 < s <= 2*N:
            sum_lengths += 2*N + 1 - s

    # Calculate lengths of forbidden i-j diagonals
    diff_lengths = 0
    for d in forbidden_iminusj:
        if d >= 0:
            diff_lengths += N - d
        else:
            diff_lengths += N + d

    # Total forbidden squares without considering overlaps
    forbidden = R * N + C * N + sum_lengths + diff_lengths

    # Calculate overlaps

    # Overlaps between rows and columns
    overlap_rows_cols = R * C

    # Overlaps between rows and i+j diagonals
    overlap_rows_sum = 0
    for r in forbidden_rows:
        for s in forbidden_iplusj:
            overlap_rows_sum += 1 if (s - r >= 1 and s - r <= N) else 0

    # Overlaps between rows and i-j diagonals
    overlap_rows_diff = 0
    for r in forbidden_rows:
        for d in forbidden_iminusj:
            c = r - d
            if 1 <= c <= N:
                overlap_rows_diff += 1

    # Overlaps between columns and i+j diagonals
    overlap_cols_sum = 0
    for c in forbidden_cols:
        for s in forbidden_iplusj:
            r = s - c
            if 1 <= r <= N:
                overlap_cols_sum += 1

    # Overlaps between columns and i-j diagonals
    overlap_cols_diff = 0
    for c in forbidden_cols:
        for d in forbidden_iminusj:
            r = c + d
            if 1 <= r <= N:
                overlap_cols_diff += 1

    # Overlaps between i+j and i-j diagonals
    overlap_sum_diff = 0
    for s in forbidden_iplusj:
        for d in forbidden_iminusj:
            r = (s + d) // 2
            c = (s - d) // 2
            if (s + d) % 2 == 0 and (s - d) % 2 == 0:
                if 1 <= r <= N and 1 <= c <= N:
                    overlap_sum_diff += 1

    # Total pairwise overlaps
    pairwise_overlaps = (
        overlap_rows_cols +
        overlap_rows_sum +
        overlap_rows_diff +
        overlap_cols_sum +
        overlap_cols_diff +
        overlap_sum_diff
    )

    # Triple overlaps

    # Overlaps among rows, columns, and i+j diagonals
    triple_overlap_rows_cols_sum = 0
    for r in forbidden_rows:
        for c in forbidden_cols:
            s = r + c
            if s in forbidden_iplusj:
                triple_overlap_rows_cols_sum += 1

    # Overlaps among rows, columns, and i-j diagonals
    triple_overlap_rows_cols_diff = 0
    for r in forbidden_rows:
        for c in forbidden_cols:
            d = r - c
            if d in forbidden_iminusj:
                triple_overlap_rows_cols_diff += 1

    # Overlaps among rows, i+j diagonals, and i-j diagonals
    triple_overlap_rows_sum_diff = 0
    for r in forbidden_rows:
        for s in forbidden_iplusj:
            c = s - r
            if 1 <= c <= N and (r - c) in forbidden_iminusj:
                triple_overlap_rows_sum_diff += 1

    # Overlaps among columns, i+j diagonals, and i-j diagonals
    triple_overlap_cols_sum_diff = 0
    for c in forbidden_cols:
        for s in forbidden_iplusj:
            r = s - c
            if 1 <= r <= N and (r - c) in forbidden_iminusj:
                triple_overlap_cols_sum_diff += 1

    # Total triple overlaps
    triple_overlaps = (
        triple_overlap_rows_cols_sum +
        triple_overlap_rows_cols_diff +
        triple_overlap_rows_sum_diff +
        triple_overlap_cols_sum_diff
    )

    # Quadruple overlaps

    # Overlaps among rows, columns, i+j diagonals, and i-j diagonals
    quadruple_overlap = 0
    for r in forbidden_rows:
        for c in forbidden_cols:
            s = r + c
            d = r - c
            if s in forbidden_iplusj and d in forbidden_iminusj:
                quadruple_overlap += 1

    # Apply inclusion-exclusion principle
    forbidden -= pairwise_overlaps
    forbidden += triple_overlaps
    forbidden -= quadruple_overlap

    # Total empty squares
    total_empty = N * N - M

    # Safe squares
    safe = total_empty - forbidden

    print(safe)

if __name__ == "__main__":
    main()