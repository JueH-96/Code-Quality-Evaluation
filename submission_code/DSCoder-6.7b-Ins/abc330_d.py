import sys

def solve(N, grid):
    rows = [0]*N
    cols = [0]*N
    o_count = [[0]*N for _ in range(N)]
    x_count = [[0]*N for _ in range(N)]

    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                o_count[i][j] = 1
                rows[i] += 1
                cols[j] += 1
            else:
                x_count[i][j] = 1

    total = 0
    for i in range(N):
        for j in range(i+1, N):
            total += (rows[i] - o_count[i][j]) * (rows[j] - o_count[j][i]) * (cols[i] - o_count[i][j]) * (cols[j] - o_count[j][i])

    return total

N = int(sys.stdin.readline().strip())
grid = [sys.stdin.readline().strip() for _ in range(N)]
print(solve(N, grid))