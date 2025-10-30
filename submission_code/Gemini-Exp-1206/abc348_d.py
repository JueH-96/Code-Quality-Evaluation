from collections import deque

H, W = map(int, input().split())
grid = [input() for _ in range(H)]
N = int(input())
medicines = []
for _ in range(N):
    r, c, e = map(int, input().split())
    medicines.append((r - 1, c - 1, e))

start = None
goal = None
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'S':
            start = (i, j)
        elif grid[i][j] == 'T':
            goal = (i, j)

q = deque()
q.append((start[0], start[1], 0, set()))
visited = set()

while q:
    x, y, energy, used_medicines = q.popleft()
    if (x, y) == goal:
        print("Yes")
        exit()

    if (x, y, energy, tuple(sorted(used_medicines))) in visited:
        continue
    visited.add((x, y, energy, tuple(sorted(used_medicines))))

    for i in range(N):
        if (x, y) == (medicines[i][0], medicines[i][1]) and i not in used_medicines:
            new_used_medicines = set(used_medicines)
            new_used_medicines.add(i)
            q.append((x, y, medicines[i][2], new_used_medicines))

    if energy > 0:
        dx = [0, 0, 1, -1]
        dy = [1, -1, 0, 0]
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
                q.append((nx, ny, energy - 1, used_medicines))

print("No")