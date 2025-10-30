import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1
forbidden = set()
offsets = [(2, 1), (2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2), (-2, 1), (-2, -1)]
for _ in range(M):
    row = int(data[index])
    index += 1
    col = int(data[index])
    index += 1
    forbidden.add((row, col))
    for dx, dy in offsets:
        nx = row + dx
        ny = col + dy
        if 1 <= nx <= N and 1 <= ny <= N:
            forbidden.add((nx, ny))
num_forbidden = len(forbidden)
total_squares = N * N
safe_squares = total_squares - num_forbidden
print(safe_squares)