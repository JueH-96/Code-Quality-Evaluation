import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1
H = int(data[index])
index += 1
K = int(data[index])
index += 1
S_str = ''.join(data[index:index + N])
index += N
items = set()
for _ in range(M):
    x = int(data[index])
    y = int(data[index + 1])
    items.add((x, y))
    index += 2
directions = {'R': (1, 0), 'L': (-1, 0), 'U': (0, 1), 'D': (0, -1)}
cx, cy = 0, 0
current_health = H
for i in range(N):
    direction_char = S_str[i]
    dx, dy = directions[direction_char]
    new_x = cx + dx
    new_y = cy + dy
    current_health -= 1
    if current_health < 0:
        print("No")
        sys.exit()
    if (new_x, new_y) in items and current_health < K:
        items.remove((new_x, new_y))
        current_health = K
    cx = new_x
    cy = new_y
print("Yes")