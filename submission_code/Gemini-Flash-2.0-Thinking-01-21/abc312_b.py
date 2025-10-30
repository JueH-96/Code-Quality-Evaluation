def solve():
    n, m = map(int, input().split())
    s = [input() for _ in range(n)]
    grid = [list(row) for row in s]
    result_pairs = []
    adjacent_cells = [
        (4, 1), (4, 2), (4, 3), (4, 4), (1, 4), (2, 4), (3, 4),
        (6, 6), (6, 7), (6, 8), (6, 9), (7, 6), (8, 6), (9, 6)
    ]
    
    for i in range(1, n - 8 + 1):
        for j in range(1, m - 8 + 1):
            is_tak_code = True
            # Check top-left 3x3 region
            for row_offset in range(3):
                for col_offset in range(3):
                    if grid[i - 1 + row_offset][j - 1 + col_offset] != '#':
                        is_tak_code = False
                        break
                if not is_tak_code:
                    break
            if not is_tak_code:
                continue
                
            # Check bottom-right 3x3 region
            for row_offset in range(3):
                for col_offset in range(3):
                    if grid[i - 1 + 6 + row_offset][j - 1 + 6 + col_offset] != '#':
                        is_tak_code = False
                        break
                if not is_tak_code:
                    break
            if not is_tak_code:
                continue
                
            # Check adjacent cells
            for r_adj, c_adj in adjacent_cells:
                if grid[i - 1 + r_adj - 1][j - 1 + c_adj - 1] != '.':
                    is_tak_code = False
                    break
            if not is_tak_code:
                continue
                
            if is_tak_code:
                result_pairs.append((i, j))
                
    for i, j in result_pairs:
        print(i, j)

if __name__ == '__main__':
    solve()