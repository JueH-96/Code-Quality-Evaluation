# YOUR CODE HERE
grid = [input().strip() for _ in range(8)]

# Initialize row and column flags
row_has_piece = [False]*8
col_has_piece = [False]*8

# Mark rows and columns which have existing pieces
for i in range(8):
    for j in range(8):
        if grid[i][j] == '#':
            row_has_piece[i] = True
            col_has_piece[j] = True

# Count the safe empty squares
count = 0
for i in range(8):
    if not row_has_piece[i]:
        for j in range(8):
            if grid[i][j] == '.' and not col_has_piece[j]:
                count +=1

print(count)