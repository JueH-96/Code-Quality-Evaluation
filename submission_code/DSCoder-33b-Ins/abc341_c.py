H, W, N = map(int, input().split())
T = input()
S = [input() for _ in range(H)]

# Initialize the current position
x, y = 0, 0
for i in range(H):
    for j in range(W):
        if S[i][j] == '.':
            x, y = i, j
            break
    else:
        continue
    break

# Define the direction vectors
dx = {'U': -1, 'D': 1, 'L': 0, 'R': 0}
dy = {'U': 0, 'D': 0, 'L': -1, 'R': 1}

# Simulate the movements
for t in T:
    nx, ny = x + dx[t], y + dy[t]
    if 0 <= nx < H and 0 <= ny < W and S[nx][ny] == '.':
        x, y = nx, ny

# Count the number of possible positions
count = 0
for i in range(H):
    for j in range(W):
        if S[i][j] == '.':
            count += 1

print(count)