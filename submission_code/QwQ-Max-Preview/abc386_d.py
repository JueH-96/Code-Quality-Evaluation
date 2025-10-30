def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1

    row_b = dict()  # For each row, store the maximum y of B cells
    row_w = dict()  # For each row, store the minimum (y-1) of W cells
    col_b = dict()  # For each column, store the maximum x of B cells
    col_w = dict()  # For each column, store the minimum (x-1) of W cells

    for _ in range(M):
        x = int(input[idx])
        idx += 1
        y = int(input[idx])
        idx += 1
        c = input[idx].strip()
        idx += 1

        # Update row dictionaries
        if c == 'B':
            if x not in row_b:
                row_b[x] = y
            else:
                row_b[x] = max(row_b[x], y)
            # Update column dictionaries for B
            if y not in col_b:
                col_b[y] = x
            else:
                col_b[y] = max(col_b[y], x)
        else:  # W
            if x not in row_w:
                row_w[x] = y - 1
            else:
                row_w[x] = min(row_w[x], y - 1)
            # Update column dictionaries for W
            if y not in col_w:
                col_w[y] = x - 1
            else:
                col_w[y] = min(col_w[y], x - 1)

    # Check for rows
    row_min = dict()
    row_max = dict()
    # Collect all rows mentioned
    rows = set(row_b.keys()).union(row_w.keys())
    for x in rows:
        r_min = row_b.get(x, 0)
        r_max = row_w.get(x, N)
        if r_min > r_max:
            print("No")
            return
        row_min[x] = r_min
        row_max[x] = r_max

    # Check for columns
    col_min = dict()
    col_max = dict()
    cols = set(col_b.keys()).union(col_w.keys())
    for y in cols:
        c_min = col_b.get(y, 0)
        c_max = col_w.get(y, N)
        if c_min > c_max:
            print("No")
            return
        col_min[y] = c_min
        col_max[y] = c_max

    # Check all B cells
    for x in row_b:
        y = row_b[x]
        if y < row_min.get(x, 0) or y > row_max.get(x, N):
            print("No")
            return
    for y in col_b:
        x = col_b[y]
        if x < col_min.get(y, 0) or x > col_max.get(y, N):
            print("No")
            return

    # Check all W cells
    for x in row_w:
        y_minus_1 = row_w[x]
        y = y_minus_1 + 1
        if y <= row_max.get(x, N) and x <= col_max.get(y, N):
            print("No")
            return
    for y in col_w:
        x_minus_1 = col_w[y]
        x = x_minus_1 + 1
        if x <= col_max.get(y, N) and y <= row_max.get(x, N):
            print("No")
            return

    print("Yes")

if __name__ == '__main__':
    main()