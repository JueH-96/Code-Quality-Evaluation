# YOUR CODE HERE
h, w = map(int, input().split())
grid = [input() for _ in range(h)]
target = "snuke"
q = [(0, 0, 0)]
visited = set()
while q:
    r, c, idx = q.pop(0)
    if (r, c) == (h - 1, w - 1):
        print("Yes")
        exit()
    if (r, c) in visited:
        continue
    visited.add((r, c))
    char = grid[r][c]
    if char == target[idx % 5]:
        next_idx = (idx + 1) % 5
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                q.append((nr, nc, next_idx))
print("No")