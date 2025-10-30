# YOUR CODE HERE
N, M = map(int, input().split())
grid = [input() for _ in range(N)]

def is_tak_code(i, j):
    if i + 8 >= N or j + 8 >= M:
        return False
    for x in range(i, i + 3):
        for y in range(j, j + 3):
            if grid[x][y] != '#':
                return False
    for x in range(i + 3, i + 6):
        for y in range(j + 3, j + 6):
            if grid[x][y] != '.':
                return False
    for x in range(i + 6, i + 9):
        for y in range(j + 6, j + 9):
            if grid[x][y] != '#':
                return False
    return True

for i in range(N):
    for j in range(M):
        if is_tak_code(i, j):
            print(i + 1, j + 1)