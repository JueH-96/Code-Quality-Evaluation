# YOUR CODE HERE
N = int(input())
grid = [input() for _ in range(N)]

row_counts = [0] * N
col_counts = [0] * N

for i in range(N):
    for j in range(N):
        if grid[i][j] == 'o':
            row_counts[i] += 1
            col_counts[j] += 1

row_pairs = sum(n * (n - 1) // 2 for n in row_counts)
col_pairs = sum(n * (n - 1) // 2 for n in col_counts)

print(row_pairs * col_pairs)