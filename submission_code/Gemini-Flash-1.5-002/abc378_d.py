H, W, K = map(int, input().split())
grid = [input() for _ in range(H)]

def is_valid(r, c):
    return 0 <= r < H and 0 <= c < W and grid[r][c] == '.'

def count_paths(r, c, path, k):
    if k == K:
        return 1
    count = 0
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if is_valid(nr, nc) and (nr, nc) not in path:
            count += count_paths(nr, nc, path | {(nr, nc)}, k + 1)
    return count

total_paths = 0
for r in range(H):
    for c in range(W):
        if grid[r][c] == '.':
            total_paths += count_paths(r, c, {(r, c)}, 0)

print(total_paths)