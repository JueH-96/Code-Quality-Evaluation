# YOUR CODE HERE
n = int(input())
grid = [input() for _ in range(n)]

# Count o's in each row and column
row_count = [0] * n
col_count = [0] * n
for i in range(n):
    for j in range(n):
        if grid[i][j] == 'o':
            row_count[i] += 1
            col_count[j] += 1

# Calculate the number of valid triples
result = 0
for i in range(n):
    for j in range(n):
        if grid[i][j] == 'o':
            result += (row_count[i] - 1) * (col_count[j] - 1)

print(result // 2)