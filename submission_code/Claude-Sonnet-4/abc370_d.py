H, W, Q = map(int, input().split())

# Initialize grid with all walls (True means wall exists)
walls = [[True for _ in range(W)] for _ in range(H)]

for _ in range(Q):
    R, C = map(int, input().split())
    R -= 1  # Convert to 0-indexed
    C -= 1  # Convert to 0-indexed
    
    if walls[R][C]:
        # If there's a wall at (R, C), destroy it
        walls[R][C] = False
    else:
        # If no wall at (R, C), look in 4 directions
        
        # Look up
        for i in range(R-1, -1, -1):
            if walls[i][C]:
                walls[i][C] = False
                break
        
        # Look down  
        for i in range(R+1, H):
            if walls[i][C]:
                walls[i][C] = False
                break
        
        # Look left
        for j in range(C-1, -1, -1):
            if walls[R][j]:
                walls[R][j] = False
                break
        
        # Look right
        for j in range(C+1, W):
            if walls[R][j]:
                walls[R][j] = False
                break

# Count remaining walls
remaining = 0
for i in range(H):
    for j in range(W):
        if walls[i][j]:
            remaining += 1

print(remaining)