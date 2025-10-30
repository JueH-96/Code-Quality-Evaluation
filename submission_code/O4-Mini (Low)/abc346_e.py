import sys
import threading

def main():
    import sys
    from collections import defaultdict

    input = sys.stdin.readline
    H, W, M = map(int, input().split())
    ops = []
    for _ in range(M):
        t, a, x = map(int, input().split())
        ops.append((t, a, x))

    # Track which rows/columns have been "accounted for" in reverse
    row_used = [False] * (H + 1)
    col_used = [False] * (W + 1)
    remaining_rows = H
    remaining_cols = W

    color_count = defaultdict(int)

    # Process operations in reverse order
    for t, a, x in reversed(ops):
        if t == 1:
            # row paint
            if not row_used[a]:
                # this row in forward painted c_rem columns
                color_count[x] += remaining_cols
                row_used[a] = True
                remaining_rows -= 1
        else:
            # column paint
            if not col_used[a]:
                # this column in forward painted r_rem rows
                color_count[x] += remaining_rows
                col_used[a] = True
                remaining_cols -= 1

    # Cells never painted by any operation remain color 0
    unpainted = remaining_rows * remaining_cols
    if unpainted > 0:
        color_count[0] += unpainted

    # Prepare output
    # Only colors with positive counts, sorted by color
    items = [(c, cnt) for c, cnt in color_count.items() if cnt > 0]
    items.sort(key=lambda x: x[0])

    print(len(items))
    for c, cnt in items:
        print(c, cnt)

if __name__ == "__main__":
    main()