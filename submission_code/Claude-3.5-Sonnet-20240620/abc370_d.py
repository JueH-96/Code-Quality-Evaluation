# YOUR CODE HERE
def find_next_wall(grid, r, c, dr, dc):
    r += dr
    c += dc
    while 0 <= r < len(grid) and 0 <= c < len(grid[0]):
        if grid[r][c]:
            return r, c
        r += dr
        c += dc
    return -1, -1

H, W, Q = map(int, input().split())
grid = [[True] * W for _ in range(H)]
remaining_walls = H * W

for _ in range(Q):
    R, C = map(int, input().split())
    R -= 1
    C -= 1
    
    if grid[R][C]:
        grid[R][C] = False
        remaining_walls -= 1
    else:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = find_next_wall(grid, R, C, dr, dc)
            if nr != -1 and nc != -1:
                grid[nr][nc] = False
                remaining_walls -= 1

print(remaining_walls)