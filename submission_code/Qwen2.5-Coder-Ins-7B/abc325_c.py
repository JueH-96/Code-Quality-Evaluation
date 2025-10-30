# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
S = data[2:]

def is_sensor(i, j):
    return S[i][j] == '#'

def is_adjacent(i1, j1, i2, j2):
    return max(abs(i1 - i2), abs(j1 - j2)) == 1

def dfs(i, j):
    stack = [(i, j)]
    while stack:
        x, y = stack.pop()
        if not is_sensor(x, y):
            continue
        S[x] = S[x][:y] + '.' + S[x][y+1:]
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and is_sensor(nx, ny):
                stack.append((nx, ny))

count = 0
for i in range(H):
    for j in range(W):
        if is_sensor(i, j):
            dfs(i, j)
            count += 1

print(count)