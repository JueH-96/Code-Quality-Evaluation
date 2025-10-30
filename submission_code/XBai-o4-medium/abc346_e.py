import bisect
from collections import defaultdict
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr]); ptr += 1
    W = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1

    row_time = [-1] * H
    row_color = [0] * H
    col_time = [-1] * W
    col_color = [0] * W

    for op_idx in range(M):
        T = int(input[ptr]); ptr += 1
        A = int(input[ptr]); ptr += 1
        X = int(input[ptr]); ptr += 1
        if T == 1:
            a = A - 1
            row_time[a] = op_idx
            row_color[a] = X
        else:
            a = A - 1
            col_time[a] = op_idx
            col_color[a] = X

    sorted_col_times = sorted(col_time)
    sorted_row_times = sorted(row_time)

    color_counts = defaultdict(int)

    # Process rows
    for i in range(H):
        rt = row_time[i]
        cnt = bisect.bisect_left(sorted_col_times, rt)
        color = row_color[i]
        color_counts[color] += cnt

    # Process columns
    for j in range(W):
        ct = col_time[j]
        cnt = bisect.bisect_left(sorted_row_times, ct)
        color = col_color[j]
        color_counts[color] += cnt

    R0 = row_time.count(-1)
    C0 = col_time.count(-1)
    color_counts[0] += R0 * C0

    # Prepare result
    res = []
    for color in color_counts:
        if color_counts[color] > 0:
            res.append((color, color_counts[color]))
    # Sort by color
    res.sort()

    # Output
    print(len(res))
    for c, cnt in res:
        print(f"{c} {cnt}")

if __name__ == "__main__":
    main()