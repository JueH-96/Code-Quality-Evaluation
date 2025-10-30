# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
grid = data[2:]

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def is_rock(i, j):
    return grid[i][j] == '#'

def is_ice(i, j):
    return grid[i][j] == '.'

def can_move(i, j, di, dj):
    while 1 <= i < N - 1 and 1 <= j < M - 1:
        if is_rock(i + di, j + dj):
            return False
        i += di
        j += dj
    return False

visited = set()
queue = [(2, 2)]
visited.add((2, 2))

while queue:
    i, j = queue.pop(0)
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 1 <= ni < N - 1 and 1 <= nj < M - 1 and (ni, nj) not in visited and can_move(i, j, di, dj):
            visited.add((ni, nj))
            queue.append((ni, nj))

print(len(visited))