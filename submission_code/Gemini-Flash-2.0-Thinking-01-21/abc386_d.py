def solve():
    n, m = map(int, input().split())
    given_cells = []
    for _ in range(m):
        x, y, c = input().split()
        given_cells.append(((int(x), int(y)), c))
    
    lower_row_c = {}
    upper_row_c = {}
    lower_col_r = {}
    upper_col_r = {}
    
    for (x, y), color in given_cells:
        if color == 'B':
            lower_row_c[x] = max(lower_row_c.get(x, 0), y)
            lower_col_r[y] = max(lower_col_r.get(y, 0), x)
        elif color == 'W':
            upper_row_c[x] = min(upper_row_c.get(x, n), y - 1)
            upper_col_r[y] = min(upper_col_r.get(y, n), x - 1)
            
    row_indices = set()
    col_indices = set()
    for (x, y), _ in given_cells:
        row_indices.add(x)
        col_indices.add(y)
        
    possible = True
    for x in row_indices:
        l_c = lower_row_c.get(x, 0)
        u_c = upper_row_c.get(x, n)
        if l_c > u_c:
            possible = False
            break
    if not possible:
        print("No")
        return
        
    for y in col_indices:
        l_r = lower_col_r.get(y, 0)
        u_r = upper_col_r.get(y, n)
        if l_r > u_r:
            possible = False
            break
    if not possible:
        print("No")
    else:
        print("Yes")

if __name__ == '__main__':
    solve()