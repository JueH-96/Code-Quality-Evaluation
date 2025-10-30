# YOUR CODE HERE
n = int(input())
grid = []
for _ in range(n):
    grid.append(input())

# Count 'o's in each row
row_count = [row.count('o') for row in grid]

# Count 'o's in each column
col_count = [sum(1 for i in range(n) if grid[i][j] == 'o') for j in range(n)]

total = 0
for i in range(n):
    for j in range(n):
        if grid[i][j] == 'o':
            # Number of L-shaped triples with (i,j) as corner
            total += (row_count[i] - 1) * (col_count[j] - 1)

print(total)