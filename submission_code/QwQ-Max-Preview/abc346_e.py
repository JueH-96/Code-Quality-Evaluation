import bisect
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr]); ptr +=1
    W = int(input[ptr]); ptr +=1
    M = int(input[ptr]); ptr +=1

    rows = [(0, 0) for _ in range(H)]
    cols = [(0, 0) for _ in range(W)]

    for step in range(1, M+1):
        T = int(input[ptr]); ptr +=1
        A = int(input[ptr]); ptr +=1
        X = int(input[ptr]); ptr +=1
        if T == 1:
            row_idx = A - 1
            rows[row_idx] = (step, X)
        else:
            col_idx = A - 1
            cols[col_idx] = (step, X)

    R = [t for t, _ in rows]
    C = [t for t, _ in cols]

    C_sorted = sorted(C)
    color_counts = defaultdict(int)

    for t, x in rows:
        cnt = bisect.bisect_left(C_sorted, t)
        color_counts[x] += cnt

    R_sorted = sorted(R)
    for t, x in cols:
        cnt = bisect.bisect_left(R_sorted, t)
        color_counts[x] += cnt

    count_rows_zero = sum(1 for t in R if t == 0)
    count_cols_zero = sum(1 for t in C if t == 0)
    count_0 = count_rows_zero * count_cols_zero
    if count_0 > 0:
        color_counts[0] += count_0

    sorted_items = sorted(color_counts.items(), key=lambda item: item[0])
    print(len(sorted_items))
    for c, cnt in sorted_items:
        print(c, cnt)

if __name__ == "__main__":
    main()