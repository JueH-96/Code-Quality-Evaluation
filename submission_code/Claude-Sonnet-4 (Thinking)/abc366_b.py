n = int(input())
strings = []
for i in range(n):
    strings.append(input())

# Find maximum length
M = max(len(s) for s in strings)

# Create matrix
matrix = [[None for _ in range(M)] for _ in range(n)]

# Fill matrix
for i in range(n):
    s = strings[i]
    # String at index i (0-based) should go to row (n-i-1) 
    # because S_{i+1} goes to position N-(i+1)+1 = n-i in 1-based indexing
    row_index = n - i - 1
    for j, char in enumerate(s):
        matrix[row_index][j] = char

# Generate output
for j in range(M):
    column = []
    for i in range(n):
        if matrix[i][j] is None:
            column.append('*')
        else:
            column.append(matrix[i][j])
    
    # Remove trailing *
    while column and column[-1] == '*':
        column.pop()
    
    print(''.join(column))