H, W, K = map(int, input().split())
grid = [input().strip() for _ in range(H)]

ans = 0

def dfs(i, j, visited, steps_left):
    if steps_left == 0:
        return 1
    count = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < H and 0 <= nj < W:
            if grid[ni][nj] == '.' and (ni, nj) not in visited:
                new_visited = visited | {(ni, nj)}
                count += dfs(ni, nj, new_visited, steps_left - 1)
    return count

for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            ans += dfs(i, j, {(i, j)}, K)

print(ans)