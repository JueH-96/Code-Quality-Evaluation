def solve():
    h_a, w_a = map(int, input().split())
    a = [input() for _ in range(h_a)]
    h_b, w_b = map(int, input().split())
    b = [input() for _ in range(h_b)]
    h_x, w_x = map(int, input().split())
    x = [input() for _ in range(h_x)]

    a_blacks = []
    for r in range(h_a):
        for c in range(w_a):
            if a[r][c] == '#':
                a_blacks.append((r, c))

    b_blacks = []
    for r in range(h_b):
        for c in range(w_b):
            if b[r][c] == '#':
                b_blacks.append((r, c))

    x_blacks = []
    for r in range(h_x):
        for c in range(w_x):
            if x[r][c] == '#':
                x_blacks.append((r, c))

    for a_row_offset in range(-min(r for r, c in a_blacks), h_x - max(r for r, c in a_blacks) - 1 + 1):
        for a_col_offset in range(-min(c for r, c in a_blacks), w_x - max(c for r, c in a_blacks) - 1 + 1):
            for b_row_offset in range(-min(r for r, c in b_blacks), h_x - max(r for r, c in b_blacks) - 1 + 1):
                for b_col_offset in range(-min(c for r, c in b_blacks), w_x - max(c for r, c in b_blacks) - 1 + 1):
                    
                    
                    
                    c = [['.' for _ in range(w_x)] for _ in range(h_x)]
                    
                    
                    all_a_blacks_covered = True
                    for r, c_ in a_blacks:
                        new_r = r + a_row_offset
                        new_c = c_ + a_col_offset
                        if 0 <= new_r < h_x and 0 <= new_c < w_x:
                            c[new_r][new_c] = '#'
                        else:
                            all_a_blacks_covered = False
                            break
                    
                    if not all_a_blacks_covered:
                        continue

                    all_b_blacks_covered = True
                    for r, c_ in b_blacks:
                        new_r = r + b_row_offset
                        new_c = c_ + b_col_offset
                        if 0 <= new_r < h_x and 0 <= new_c < w_x:
                            c[new_r][new_c] = '#'
                        else:
                            all_b_blacks_covered = False
                            break
                    
                    if not all_b_blacks_covered:
                        continue
                    
                    
                    all_blacks_from_a_and_b_covered = True
                    for r, c_ in a_blacks:
                        new_r = r + a_row_offset
                        new_c = c_ + a_col_offset
                        if c[new_r][new_c] != '#':
                            all_blacks_from_a_and_b_covered = False
                            break
                    if not all_blacks_from_a_and_b_covered:
                        continue
                    
                    all_blacks_from_a_and_b_covered = True
                    for r, c_ in b_blacks:
                        new_r = r + b_row_offset
                        new_c = c_ + b_col_offset
                        if c[new_r][new_c] != '#':
                            all_blacks_from_a_and_b_covered = False
                            break
                    if not all_blacks_from_a_and_b_covered:
                        continue
                    
                    
                    
                    match = True
                    for r in range(h_x):
                        for c_ in range(w_x):
                            if (x[r][c_] == '#' and c[r][c_] != '#') or (x[r][c_] == '.' and c[r][c_] != '.'):
                                match = False
                                break
                        if not match:
                            break
                    
                    if match:
                        print("Yes")
                        return

    print("No")

solve()