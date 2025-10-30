import bisect

def main():
    import sys
    H, W, M = map(int, sys.stdin.readline().split())

    row_time = [0] * (H + 1)  # 1-based indexing
    row_color = [0] * (H + 1)
    col_time = [0] * (W + 1)
    col_color = [0] * (W + 1)

    for step in range(1, M + 1):
        T, A, X = map(int, sys.stdin.readline().split())
        if T == 1:
            r = A
            row_time[r] = step
            row_color[r] = X
        else:
            c = A
            col_time[c] = step
            col_color[c] = X

    # Collect row and column times
    row_times = [row_time[r] for r in range(1, H + 1)]
    col_times = [col_time[c] for c in range(1, W + 1)]

    sorted_col_times = sorted(col_times)
    sorted_row_times = sorted(row_times)

    from collections import defaultdict
    counts = defaultdict(int)

    # Process each row
    for r in range(1, H + 1):
        rt = row_time[r]
        rc = row_color[r]
        cnt = bisect.bisect_left(sorted_col_times, rt)
        counts[rc] += cnt

    # Process each column
    for c in range(1, W + 1):
        ct = col_time[c]
        cc = col_color[c]
        cnt = bisect.bisect_left(sorted_row_times, ct)
        counts[cc] += cnt

    # Calculate additional cells where both row and column were never painted
    count_r0 = row_times.count(0)
    count_c0 = col_times.count(0)
    additional_0 = count_r0 * count_c0
    counts[0] += additional_0

    # Prepare the result
    result = []
    for color in counts:
        if counts[color] > 0:
            result.append((color, counts[color]))
    result.sort()

    # Output
    print(len(result))
    for c, cnt in result:
        print(c, cnt)

if __name__ == "__main__":
    main()