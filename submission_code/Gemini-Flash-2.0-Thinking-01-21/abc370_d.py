def solve():
    h, w, q = map(int, input().split())
    queries = []
    for _ in range(q):
        queries.append(tuple(map(int, input().split())))
    
    walls = [[True for _ in range(w)] for _ in range(h)]
    
    for r_q, c_q in queries:
        r_q -= 1 # 0-indexed row
        c_q -= 1 # 0-indexed column
        
        if walls[r_q][c_q]:
            walls[r_q][c_q] = False
        else:
            # Upward direction
            destroyed_up = False
            for i in range(r_q - 1, -1, -1):
                if walls[i][c_q]:
                    is_clear_path = True
                    for k in range(i + 1, r_q):
                        if walls[k][c_q]:
                            is_clear_path = False
                            break
                    if is_clear_path:
                        walls[i][c_q] = False
                        destroyed_up = True
                        break
                        
            # Downward direction
            destroyed_down = False
            for i in range(r_q + 1, h):
                if walls[i][c_q]:
                    is_clear_path = True
                    for k in range(r_q + 1, i):
                        if walls[k][c_q]:
                            is_clear_path = False
                            break
                    if is_clear_path:
                        walls[i][c_q] = False
                        destroyed_down = True
                        break
                        
            # Leftward direction
            destroyed_left = False
            for j in range(c_q - 1, -1, -1):
                if walls[r_q][j]:
                    is_clear_path = True
                    for k in range(j + 1, c_q):
                        if walls[r_q][k]:
                            is_clear_path = False
                            break
                    if is_clear_path:
                        walls[r_q][j] = False
                        destroyed_left = True
                        break
                        
            # Rightward direction
            destroyed_right = False
            for j in range(c_q + 1, w):
                if walls[r_q][j]:
                    is_clear_path = True
                    for k in range(c_q + 1, j):
                        if walls[r_q][k]:
                            is_clear_path = False
                            break
                    if is_clear_path:
                        walls[r_q][j] = False
                        destroyed_right = True
                        break
                        
    remaining_walls_count = 0
    for i in range(h):
        for j in range(w):
            if walls[i][j]:
                remaining_walls_count += 1
                
    print(remaining_walls_count)

if __name__ == '__main__':
    solve()