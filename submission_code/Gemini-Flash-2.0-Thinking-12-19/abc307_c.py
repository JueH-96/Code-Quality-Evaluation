def solve():
    h_a, w_a = map(int, input().split())
    a_grid = [input() for _ in range(h_a)]
    h_b, w_b = map(int, input().split())
    b_grid = [input() for _ in range(h_b)]
    h_x, w_x = map(int, input().split())
    x_grid = [input() for _ in range(h_x)]
    
    black_squares_a = []
    for i in range(h_a):
        for j in range(w_a):
            if a_grid[i][j] == '#':
                black_squares_a.append((i, j))
                
    black_squares_b = []
    for i in range(h_b):
        for j in range(w_b):
            if b_grid[i][j] == '#':
                black_squares_b.append((i, j))
                
    black_squares_x = []
    for i in range(h_x):
        for j in range(w_x):
            if x_grid[i][j] == '#':
                black_squares_x.append((i, j))
                
    for delta_r in range(-(h_b - 1), h_a):
        for delta_c in range(-(w_b - 1), w_a):
            s_a = set(black_squares_a)
            s_b = set()
            for r_b, c_b in black_squares_b:
                s_b.add((r_b + delta_r, c_b + delta_c))
            s = s_a.union(s_b)
            if not s:
                continue
                
            min_r = min(r for r, c in s)
            max_r = max(r for r, c in s)
            min_c = min(c for r, c in s)
            max_c = max(c for r, c in s)
            
            h = max_r - min_r + 1
            w = max_c - min_c + 1
            
            if h > h_x or w > w_x:
                continue
                
            start_row = max(0, max_r - h_x + 1)
            end_row = min_r
            start_col = max(0, max_c - w_x + 1)
            end_col = min_c
            
            if start_row > end_row or start_col > end_col:
                continue
                
            for cut_row in range(start_row, end_row + 1):
                for cut_col in range(start_col, end_col + 1):
                    x_prime_grid = [['.' for _ in range(w_x)] for _ in range(h_x)]
                    for r, c in s:
                        if cut_row <= r < cut_row + h_x and cut_col <= c < cut_col + w_x:
                            x_prime_grid[r - cut_row][c - cut_col] = '#'
                            
                    x_prime_lines = ["".join(row) for row in x_prime_grid]
                    if x_prime_lines == x_grid:
                        print("Yes")
                        return
                        
    print("No")

if __name__ == '__main__':
    solve()