n = int(input())
P = list(map(int, input().split()))
Q = list(map(int, input().split()))

# Create pos_row where pos_row[i] is the position of row i in P
pos_row = [0] * (n + 1)  # 1-based indexing
for idx, val in enumerate(P):
    pos_row[val] = idx + 1  # since P is 1-based in the problem statement

# Create pos_col where pos_col[j] is the position of column j in Q
pos_col = [0] * (n + 1)
for idx, val in enumerate(Q):
    pos_col[val] = idx + 1

# Fill the grid
grid = []
for i in range(1, n + 1):
    row = []
    for j in range(1, n + 1):
        if pos_row[i] < pos_col[j]:
            row.append('0')
        else:
            row.append('1')
    grid.append(''.join(row))

# Print the grid
for line in grid:
    print(line)