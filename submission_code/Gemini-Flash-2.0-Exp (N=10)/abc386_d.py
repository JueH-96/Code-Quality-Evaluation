def solve():
    n, m = map(int, input().split())
    colored = []
    for _ in range(m):
        x, y, c = input().split()
        colored.append((int(x), int(y), c))

    row_black_limit = [0] * (n + 1)
    col_black_limit = [0] * (n + 1)
    
    for x, y, c in colored:
        if c == 'B':
            row_black_limit[x] = max(row_black_limit[x], y)
            col_black_limit[y] = max(col_black_limit[y], x)
        else:
            if row_black_limit[x] >= y:
                print("No")
                return
            if col_black_limit[y] >= x:
                print("No")
                return

    max_row_black = 0
    for i in range(1, n + 1):
      max_row_black = max(max_row_black, row_black_limit[i])
    
    max_col_black = 0
    for i in range(1, n + 1):
      max_col_black = max(max_col_black, col_black_limit[i])

    
    for x, y, c in colored:
        if c == 'B':
            if y > max_row_black:
                print("No")
                return
            if x > max_col_black:
                print("No")
                return
        else:
            if y <= max_row_black and x <= max_col_black:
                print("No")
                return
    
    print("Yes")

solve()