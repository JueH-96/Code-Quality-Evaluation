H, W, K = map(int, input().split())
grid = [input().strip() for _ in range(H)]

dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
adj = [[[] for _ in range(W)] for _ in range(H)]
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.':
                    adj[i][j].append((ni, nj))

def backtrack(i, j, mask, steps):
    if steps == K:
        return 1
    res = 0
    for ni, nj in adj[i][j]:
        nidx = ni * W + nj
        if not (mask & (1 << nidx)):
            res += backtrack(ni, nj, mask | (1 << nidx), steps + 1)
    return res

total = 0
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            idx = i * W + j
            total += backtrack(i, j, 1 << idx, 0)

print(total)