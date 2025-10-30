def solve():
    n, m = map(int, input().split())
    colored_cells = []
    for _ in range(m):
        colored_cells.append(list(input().split()))

    rows = {}
    cols = {}

    for x, y, c in colored_cells:
        x = int(x)
        y = int(y)
        if x not in rows:
            rows[x] = []
        rows[x].append((y, c))
        if y not in cols:
            cols[y] = []
        cols[y].append((x, c))

    for row in rows:
        rows[row].sort()
        for i in range(len(rows[row]) - 1):
            if rows[row][i][1] == 'W' and rows[row][i+1][1] == 'B':
                print("No")
                return

    for col in cols:
        cols[col].sort()
        for i in range(len(cols[col]) - 1):
            if cols[col][i][1] == 'W' and cols[col][i+1][1] == 'B':
                print("No")
                return
    
    row_limits = {}
    col_limits = {}

    for row_num, cells in rows.items():
        max_black = 0
        min_white = n + 1
        for col_num, color in cells:
            if color == 'B':
                max_black = max(max_black, col_num)
            else:
                min_white = min(min_white, col_num)
        row_limits[row_num] = (max_black, min_white)

    for col_num, cells in cols.items():
        max_black = 0
        min_white = n + 1
        for row_num, color in cells:
            if color == 'B':
                max_black = max(max_black, row_num)
            else:
                min_white = min(min_white, row_num)
        col_limits[col_num] = (max_black, min_white)

    
    sorted_rows = sorted(row_limits.keys())
    for i in range(len(sorted_rows) - 1):
        if row_limits[sorted_rows[i]][0] > row_limits[sorted_rows[i+1]][0]:
            print("No")
            return
        if row_limits[sorted_rows[i]][1] < row_limits[sorted_rows[i+1]][1]:
            print("No")
            return

    sorted_cols = sorted(col_limits.keys())
    for i in range(len(sorted_cols) - 1):
        if col_limits[sorted_cols[i]][0] > col_limits[sorted_cols[i+1]][0]:
            print("No")
            return
        if col_limits[sorted_cols[i]][1] < col_limits[sorted_cols[i+1]][1]:
            print("No")
            return

    print("Yes")

solve()