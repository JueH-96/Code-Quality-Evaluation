H, W, Q = map(int, input().split())
walls = [[True] * W for _ in range(H)]

for _ in range(Q):
    r, c = map(int, input().split())
    r -= 1
    c -= 1
    
    if walls[r][c]:
        walls[r][c] = False
    else:
        # Look up
        for i in range(r-1, -1, -1):
            if walls[i][c]:
                walls[i][c] = False
                break
                
        # Look down
        for i in range(r+1, H):
            if walls[i][c]:
                walls[i][c] = False
                break
                
        # Look left
        for j in range(c-1, -1, -1):
            if walls[r][j]:
                walls[r][j] = False
                break
                
        # Look right
        for j in range(c+1, W):
            if walls[r][j]:
                walls[r][j] = False
                break

ans = sum(sum(row) for row in walls)
print(ans)