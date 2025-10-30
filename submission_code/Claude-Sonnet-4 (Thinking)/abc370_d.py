H, W, Q = map(int, input().split())

walls = [[True] * W for _ in range(H)]

for _ in range(Q):
    r, c = map(int, input().split())
    r -= 1  # Convert to 0-indexed
    c -= 1  # Convert to 0-indexed
    
    if walls[r][c]:
        walls[r][c] = False
    else:
        # Up direction
        for i in range(r-1, -1, -1):
            if walls[i][c]:
                walls[i][c] = False
                break
        
        # Down direction
        for i in range(r+1, H):
            if walls[i][c]:
                walls[i][c] = False
                break
        
        # Left direction
        for j in range(c-1, -1, -1):
            if walls[r][j]:
                walls[r][j] = False
                break
        
        # Right direction
        for j in range(c+1, W):
            if walls[r][j]:
                walls[r][j] = False
                break

count = sum(sum(row) for row in walls)
print(count)