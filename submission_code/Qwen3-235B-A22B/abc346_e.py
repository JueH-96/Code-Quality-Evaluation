import bisect
from collections import defaultdict
import sys

def main():
    H, W, M = map(int, sys.stdin.readline().split())
    row_time = [-1] * H
    row_color = [0] * H
    col_time = [-1] * W
    col_color = [0] * W

    for time in range(M):
        parts = sys.stdin.readline().split()
        T = int(parts[0])
        A = int(parts[1])
        X = int(parts[2])
        if T == 1:
            r = A - 1
            row_time[r] = time
            row_color[r] = X
        else:
            c = A - 1
            col_time[c] = time
            col_color[c] = X

    # Build sorted lists of times
    sorted_rows = sorted(row_time)
    sorted_cols = sorted(col_time)

    color_counts = defaultdict(int)

    # Process each row
    for i in range(H):
        t = row_time[i]
        if t == -1:
            continue
        color = row_color[i]
        cnt = bisect.bisect_left(sorted_cols, t)
        if cnt > 0:
            color_counts[color] += cnt

    # Process each column
    for i in range(W):
        t = col_time[i]
        color = col_color[i]
        cnt = bisect.bisect_right(sorted_rows, t)
        if cnt > 0:
            color_counts[color] += cnt

    # Prepare the result
    result = []
    for color in color_counts:
        if color_counts[color] > 0:
            result.append((color, color_counts[color]))
    result.sort()

    # Output
    print(len(result))
    for color, count in result:
        print(f"{color} {count}")

if __name__ == "__main__":
    main()