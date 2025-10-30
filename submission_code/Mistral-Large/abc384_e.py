import sys
from heapq import heappop, heappush

input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
X = int(data[2])
P = int(data[3]) - 1
Q = int(data[4]) - 1

S = []
index = 5
for i in range(H):
    row = []
    for j in range(W):
        row.append(int(data[index]))
        index += 1
    S.append(row)

strength = S[P][Q]
visited = [[False] * W for _ in range(H)]
visited[P][Q] = True

heap = []
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

for dx, dy in directions:
    nx, ny = P + dx, Q + dy
    if 0 <= nx < H and 0 <= ny < W:
        heappush(heap, (-S[nx][ny], nx, ny))

while heap:
    current_strength, x, y = heappop(heap)
    current_strength = -current_strength

    if visited[x][y] or current_strength >= strength / X:
        continue

    strength += current_strength
    visited[x][y] = True

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny]:
            heappush(heap, (-S[nx][ny], nx, ny))

print(strength)