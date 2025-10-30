n = int(input())
p = list(map(int, input().split()))
q = list(map(int, input().split()))

# Convert to 0-indexed
p = [x - 1 for x in p]
q = [x - 1 for x in q]

# Create the grid
grid = [[0] * n for _ in range(n)]

# Fill the grid based on the relative positions in permutations
for i in range(n):
    for j in range(n):
        # Find positions of row i and column j in their respective permutations
        row_pos = p.index(i)
        col_pos = q.index(j)
        
        # Set grid[i][j] = 1 if row_pos > col_pos, else 0
        if row_pos > col_pos:
            grid[i][j] = 1
        else:
            grid[i][j] = 0

# Print the result
for i in range(n):
    print(''.join(map(str, grid[i])))