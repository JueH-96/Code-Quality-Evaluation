def solve():
    h, w, q = map(int, input().split())
    queries = []
    for _ in range(q):
        queries.append(tuple(map(int, input().split())))
    
    walls = [[True for _ in range(w)] for _ in range(h)]
    
    for r_q, c_q in queries:
        r, c = r_q - 1, c_q - 1
        if walls[r][c]:
            walls[r][c] = False
        else:
            # Up direction
            target_row_up = -1
            for i in range(r - 1, -1, -1):
                if walls[i][c]:
                    is_clear_path = True
                    for k in range(i + 1, r):
                        if walls[k][c]:
                            is_clear_path = False
                            break
                    if is_clear_path:
                        target_row_up = i
                        break
            if target_row_up != -1:
                walls[target_row_up][c] = False
                
            # Down direction
            target_row_down = -1
            for i in range(r + 1, h):
                if walls[i][c]:
                    is_clear_path = True
                    for k in range(r + 1, i):
                        if walls[k][c]:
                            is_clear_path = False
                            break
                    if is_clear_path:
                        target_row_down = i
                        break
            if target_row_down != -1:
                walls[target_row_down][c] = False
                
            # Left direction
            target_col_left = -1
            for j in range(c - 1, -1, -1):
                if walls[r][j]:
                    is_clear_path = True
                    for k in range(j + 1, c):
                        if walls[r][k]:
                            is_clear_path = False
                            break
                    if is_clear_path:
                        target_col_left = j
                        break
            if target_col_left != -1:
                walls[r][target_col_left] = False
                
            # Right direction
            target_col_right = -1
            for j in range(c + 1, w):
                if walls[r][j]:
                    is_clear_path = True
                    for k in range(c + 1, j):
                        if walls[r][k]:
                            is_clear_path = False
                            break
                    if is_clear_path:
                        target_col_right = j
                        break
            if target_col_right != -1:
                walls[r][target_col_right] = False
                
    remaining_walls_count = 0
    for i in range(h):
        for j in range(w):
            if walls[i][j]:
                remaining_walls_count += 1
                
    print(remaining_walls_count)

if __name__ == '__main__':
    solve()