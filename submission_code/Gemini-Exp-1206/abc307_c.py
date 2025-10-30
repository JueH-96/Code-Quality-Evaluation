def solve():
    h_a, w_a = map(int, input().split())
    a = [input() for _ in range(h_a)]
    h_b, w_b = map(int, input().split())
    b = [input() for _ in range(h_b)]
    h_x, w_x = map(int, input().split())
    x = [input() for _ in range(h_x)]

    def get_black_squares(grid, h, w):
        black_squares = []
        for r in range(h):
            for c in range(w):
                if grid[r][c] == '#':
                    black_squares.append((r, c))
        return black_squares

    black_squares_a = get_black_squares(a, h_a, w_a)
    black_squares_b = get_black_squares(b, h_b, w_b)
    black_squares_x = get_black_squares(x, h_x, w_x)

    for start_row_a in range(-h_a + 1, h_x):
        for start_col_a in range(-w_a + 1, w_x):
            for start_row_b in range(-h_b + 1, h_x):
                for start_col_b in range(-w_b + 1, w_x):
                    
                    combined_grid = [['.' for _ in range(w_x)] for _ in range(h_x)]
                    
                    all_black_included = True
                    
                    for r_a, c_a in black_squares_a:
                        r = start_row_a + r_a
                        c = start_col_a + c_a
                        if 0 <= r < h_x and 0 <= c < w_x:
                            combined_grid[r][c] = '#'
                        else:
                            all_black_included = False
                            
                    for r_b, c_b in black_squares_b:
                        r = start_row_b + r_b
                        c = start_col_b + c_b
                        if 0 <= r < h_x and 0 <= c < w_x:
                            combined_grid[r][c] = '#'
                        else:
                            all_black_included = False

                    if not all_black_included:
                        continue
                    
                    match = True
                    for r in range(h_x):
                        for c in range(w_x):
                            if combined_grid[r][c] != x[r][c]:
                                match = False
                                break
                        if not match:
                            break
                    
                    if match:
                        print("Yes")
                        return

    print("No")

solve()