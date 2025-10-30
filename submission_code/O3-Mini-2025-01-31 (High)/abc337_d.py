def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    H = int(data[0])
    W = int(data[1])
    K = int(data[2])
    grid = data[3:3 + H]

    # We want to obtain a contiguous segment of length K (either horizontally in a row or vertically in a column)
    # that does not contain any cell with an 'x'. We are allowed to convert cells with '.' into 'o', each
    # conversion costing 1 operation. The aim is to minimize the total number of conversions.
    #
    # Thus, for a candidate segment, if it contains any 'x' (which cannot be converted), then it is not valid.
    # Otherwise, the cost is simply the number of dots in the segment.
    #
    # We'll examine every possible horizontal segment in each row and the vertical segments in each column,
    # using prefix sums to count the number of dots and x's quickly in each segment.

    INF = 10**9
    best = INF

    # Check horizontal segments
    for i in range(H):
        row = grid[i]
        if W < K:
            continue  # There is no horizontal segment of length K in this row.
        dot_prefix = [0] * (W + 1)
        x_prefix = [0] * (W + 1)
        for j in range(W):
            dot_prefix[j + 1] = dot_prefix[j] + (1 if row[j] == '.' else 0)
            x_prefix[j + 1] = x_prefix[j] + (1 if row[j] == 'x' else 0)
        for start in range(W - K + 1):
            # If this segment contains no 'x', it's a candidate.
            if x_prefix[start + K] - x_prefix[start] == 0:
                cost = dot_prefix[start + K] - dot_prefix[start]
                if cost < best:
                    best = cost

    # Check vertical segments
    for j in range(W):
        if H < K:
            continue  # There is no vertical segment of length K in this column.
        # Build the j-th column as a list of characters.
        col = [grid[i][j] for i in range(H)]
        dot_prefix = [0] * (H + 1)
        x_prefix = [0] * (H + 1)
        for i in range(H):
            dot_prefix[i + 1] = dot_prefix[i] + (1 if col[i] == '.' else 0)
            x_prefix[i + 1] = x_prefix[i] + (1 if col[i] == 'x' else 0)
        for start in range(H - K + 1):
            if x_prefix[start + K] - x_prefix[start] == 0:
                cost = dot_prefix[start + K] - dot_prefix[start]
                if cost < best:
                    best = cost

    if best == INF:
        sys.stdout.write("-1")
    else:
        sys.stdout.write(str(best))

if __name__ == '__main__':
    main()