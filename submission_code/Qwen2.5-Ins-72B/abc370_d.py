# YOUR CODE HERE
H, W, Q = map(int, input().split())
walls = [[True] * W for _ in range(H)]

def destroy_walls(r, c):
    if walls[r][c]:
        walls[r][c] = False
        return 1
    else:
        destroyed = 0
        # Up
        for i in range(r-1, -1, -1):
            if walls[i][c]:
                walls[i][c] = False
                destroyed += 1
                break
        # Down
        for i in range(r+1, H):
            if walls[i][c]:
                walls[i][c] = False
                destroyed += 1
                break
        # Left
        for j in range(c-1, -1, -1):
            if walls[r][j]:
                walls[r][j] = False
                destroyed += 1
                break
        # Right
        for j in range(c+1, W):
            if walls[r][j]:
                walls[r][j] = False
                destroyed += 1
                break
        return destroyed

total_walls = H * W
for _ in range(Q):
    R, C = map(int, input().split())
    total_walls -= destroy_walls(R-1, C-1)

print(total_walls)