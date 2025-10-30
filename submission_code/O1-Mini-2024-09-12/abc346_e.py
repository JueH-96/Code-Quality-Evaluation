import sys, bisect
from collections import defaultdict

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    H, W, M = map(int, sys.stdin.readline().split())
    row_last = [0] * H
    row_color = [0] * H
    col_last = [0] * W
    col_color = [0] * W
    for i in range(1, M +1):
        parts = sys.stdin.readline().split()
        T, A, X = int(parts[0]), int(parts[1]), int(parts[2])
        if T ==1:
            row_last[A-1] = i
            row_color[A-1] = X
        else:
            col_last[A-1] = i
            col_color[A-1] = X
    # Prepare sorted column last times
    sorted_col_times = sorted(col_last)
    # Prepare sorted row last times
    sorted_row_times = sorted(row_last)
    color_to_count = defaultdict(int)
    # Process rows
    for r in range(H):
        t = row_last[r]
        if t >0:
            cnt = bisect.bisect_left(sorted_col_times, t)
            color = row_color[r]
            color_to_count[color] += cnt
    # Process columns
    for c in range(W):
        t = col_last[c]
        if t >0:
            cnt = bisect.bisect_left(sorted_row_times, t)
            color = col_color[c]
            color_to_count[color] += cnt
    # Cells not updated by any row or column
    total_assigned = sum(color_to_count.values())
    remaining = H * W - total_assigned
    color_to_count[0] += remaining
    # Prepare output
    items = sorted((k, v) for k, v in color_to_count.items() if v >0)
    K = len(items)
    print(K)
    for k, v in items:
        print(k, v)

if __name__ == "__main__":
    main()