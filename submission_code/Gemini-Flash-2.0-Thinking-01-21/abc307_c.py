def solve():
    h_a, w_a = map(int, input().split())
    sheet_a = [input() for _ in range(h_a)]
    h_b, w_b = map(int, input().split())
    sheet_b = [input() for _ in range(h_b)]
    h_x, w_x = map(int, input().split())
    sheet_x = [input() for _ in range(h_x)]
    
    black_a_squares = []
    for i in range(h_a):
        for j in range(w_a):
            if sheet_a[i][j] == '#':
                black_a_squares.append((i, j))
    black_b_squares = []
    for i in range(h_b):
        for j in range(w_b):
            if sheet_b[i][j] == '#':
                black_b_squares.append((i, j))
    black_x_squares = []
    for i in range(h_x):
        for j in range(w_x):
            if sheet_x[i][j] == '#':
                black_x_squares.append((i, j))
                
    r_a_min, r_a_max = -float('inf'), float('inf')
    c_a_min, c_a_max = -float('inf'), float('inf')
    if black_a_squares:
        r_a_min_init = max([-r for r, c in black_a_squares])
        r_a_max_init = min([h_x - 1 - r for r, c in black_a_squares])
        c_a_min_init = max([-c for r, c in black_a_squares])
        c_a_max_init = min([w_x - 1 - c for r, c in black_a_squares])
        if r_a_min_init > r_a_max_init or c_a_min_init > c_a_max_init:
            print("No")
            return
        r_a_min, r_a_max = r_a_min_init, r_a_max_init
        c_a_min, c_a_max = c_a_min_init, c_a_max_init
    else:
        r_a_min, r_a_max = 0, 0
        c_a_min, c_a_max = 0, 0

    r_b_min, r_b_max = -float('inf'), float('inf')
    c_b_min, c_b_max = -float('inf'), float('inf')
    if black_b_squares:
        r_b_min_init = max([-r for r, c in black_b_squares])
        r_b_max_init = min([h_x - 1 - r for r, c in black_b_squares])
        c_b_min_init = max([-c for r, c in black_b_squares])
        c_b_max_init = min([w_x - 1 - c for r, c in black_b_squares])
        if r_b_min_init > r_b_max_init or c_b_min_init > c_b_max_init:
            print("No")
            return
        r_b_min, r_b_max = r_b_min_init, r_b_max_init
        c_b_min, c_b_max = c_b_min_init, c_b_max_init
    else:
        r_b_min, r_b_max = 0, 0
        c_b_min, c_b_max = 0, 0

    for r_a in range(r_a_min, r_a_max + 1):
        for c_a in range(c_a_min, c_a_max + 1):
            for r_b in range(r_b_min, r_b_max + 1):
                for c_b in range(c_b_min, c_b_max + 1):
                    current_sheet = [['.' for _ in range(w_x)] for _ in range(h_x)]
                    for i, j in black_a_squares:
                        current_sheet[r_a + i][c_a + j] = '#'
                    for i, j in black_b_squares:
                        current_sheet[r_b + i][c_b + j] = '#'
                        
                    is_match = True
                    for i in range(h_x):
                        for j in range(w_x):
                            if current_sheet[i][j] != sheet_x[i][j]:
                                is_match = False
                                break
                        if not is_match:
                            break
                    if is_match:
                        print("Yes")
                        return
                        
    print("No")

if __name__ == '__main__':
    solve()