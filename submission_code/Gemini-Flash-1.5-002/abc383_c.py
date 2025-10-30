import collections

H, W, D = map(int, input().split())
grid = [list(input()) for _ in range(H)]

humidifiers = []
for r in range(H):
    for c in range(W):
        if grid[r][c] == 'H':
            humidifiers.append((r, c))

q = collections.deque(humidifiers)
visited = set(humidifiers)
humidified_count = len(humidifiers)

while q:
    r, c = q.popleft()
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        dist = 1
        while 0 <= nr < H and 0 <= nc < W and dist <= D:
            if grid[nr][nc] == '#' :
                break
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                humidified_count +=1
                q.append((nr,nc))
            nr += dr
            nc += dc
            dist +=1

print(humidified_count)