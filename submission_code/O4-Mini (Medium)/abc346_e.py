import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    h = int(data[0])
    w = int(data[1])
    m = int(data[2])
    # Read operations
    ops = []
    idx = 3
    for _ in range(m):
        t = int(data[idx]); a = int(data[idx+1]); x = int(data[idx+2])
        ops.append((t, a-1, x))  # zero-index the row/col
        idx += 3

    # Track which rows/columns have been painted (in reverse)
    row_painted = [False] * h
    col_painted = [False] * w
    rem_rows = h
    rem_cols = w

    from collections import defaultdict
    color_count = defaultdict(int)

    # Process operations in reverse
    for t, a, x in reversed(ops):
        if t == 1:
            # paint row a
            if not row_painted[a]:
                row_painted[a] = True
                # this paints all currently unpainted columns in this row
                if rem_cols > 0:
                    color_count[x] += rem_cols
                rem_rows -= 1
        else:
            # paint column a
            if not col_painted[a]:
                col_painted[a] = True
                if rem_rows > 0:
                    color_count[x] += rem_rows
                rem_cols -= 1

    # The cells never painted by any op stay color 0
    # their count = rem_rows * rem_cols
    zero_cells = rem_rows * rem_cols
    if zero_cells > 0:
        color_count[0] += zero_cells

    # Prepare output: only colors with positive count
    items = [(c, cnt) for c, cnt in color_count.items() if cnt > 0]
    items.sort(key=lambda x: x[0])

    # Print
    out = []
    out.append(str(len(items)))
    for c, cnt in items:
        out.append(f"{c} {cnt}")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()