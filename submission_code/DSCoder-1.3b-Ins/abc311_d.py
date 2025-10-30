N, M = map(int, input().split())
grid = [input() for _ in range(N)]

def count_ice(i, j):
    if i < 1 or i >= N or j < 1 or j >= M:
        return 0
    if grid[i][j] == '#':
        return 0
    grid[i][j] = '#'
    return 1 + count_ice(i-1, j) + count_ice(i+1, j) + count_ice(i, j-1) + count_ice(i, j+1)

print(count_ice(2, 2))