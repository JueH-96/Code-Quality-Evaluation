import sys

def count_triples(N, grid):
    count = 0
    for i in range(N):
        for j in range(N):
            for k in range(j+1, N):
                if grid[i][j] == 'o' and grid[i][k] == 'o' and grid[j][i] == 'o':
                    count += 1
                if grid[j][i] == 'o' and grid[k][i] == 'o' and grid[i][j] == 'o':
                    count += 1
    return count

N = int(sys.stdin.readline().strip())
grid = [sys.stdin.readline().strip() for _ in range(N)]
print(count_triples(N, grid))