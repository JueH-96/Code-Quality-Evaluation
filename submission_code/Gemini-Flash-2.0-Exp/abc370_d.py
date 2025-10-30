def solve():
    H, W, Q = map(int, input().split())
    
    walls = [[True for _ in range(W)] for _ in range(H)]
    
    def destroy_wall(r, c):
        walls[r][c] = False
    
    def find_and_destroy(r, c):
        # Up
        for i in range(r - 1, -1, -1):
            wall_found = True
            for k in range(i + 1, r):
                if not walls[k][c]:
                    continue
                else:
                    wall_found = False
                    break
            if wall_found and walls[i][c]:
                destroy_wall(i, c)
                break
        
        # Down
        for i in range(r + 1, H):
            wall_found = True
            for k in range(r + 1, i):
                if not walls[k][c]:
                    continue
                else:
                    wall_found = False
                    break
            if wall_found and walls[i][c]:
                destroy_wall(i, c)
                break
        
        # Left
        for j in range(c - 1, -1, -1):
            wall_found = True
            for k in range(j + 1, c):
                if not walls[r][k]:
                    continue
                else:
                    wall_found = False
                    break
            if wall_found and walls[r][j]:
                destroy_wall(r, j)
                break
        
        # Right
        for j in range(c + 1, W):
            wall_found = True
            for k in range(c + 1, j):
                if not walls[r][k]:
                    continue
                else:
                    wall_found = False
                    break
            if wall_found and walls[r][j]:
                destroy_wall(r, j)
                break
    
    for _ in range(Q):
        R, C = map(int, input().split())
        R -= 1
        C -= 1
        
        if walls[R][C]:
            destroy_wall(R, C)
        else:
            find_and_destroy(R, C)
    
    remaining_walls = 0
    for i in range(H):
        for j in range(W):
            if walls[i][j]:
                remaining_walls += 1
    
    print(remaining_walls)

solve()