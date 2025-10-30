n = int(input())
grid = [input().strip() for _ in range(n)]

# Calculate the number of 'o's in each row
row_counts = [row.count('o') for row in grid]

# Calculate the number of 'o's in each column
col_counts = [0] * n
for j in range(n):
    for i in range(n):
        if grid[i][j] == 'o':
            col_counts[j] += 1

total = 0
for i in range(n):
    for j in range(n):
        if grid[i][j] == 'o':
            total += (row_counts[i] - 1) * (col_counts[j] - 1)

print(total)