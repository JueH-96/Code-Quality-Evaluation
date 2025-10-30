N = int(input())
grid = [list(input()) for _ in range(N)]

def solve():
    row_counts = [0] * N
    col_counts = [0] * N
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                row_counts[i] += 1
                col_counts[j] += 1

    ans = 0
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                ans += row_counts[i] * (row_counts[i] - 1) // 2
                ans += col_counts[j] * (col_counts[j] - 1) // 2
                ans -= (row_counts[i] - 1) * (col_counts[j] - 1)
    return ans // 2

print(solve())