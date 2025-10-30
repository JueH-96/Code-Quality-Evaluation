def get_cell(n, i, j):
    # Base case: level 0 is always black
    if n == 0:
        return '#'
    
    # Size of each block at this level
    size = 3 ** (n - 1)
    
    # Determine which block this cell belongs to
    block_i = i // size
    block_j = j // size
    
    # Central block (1, 1) is always white
    if block_i == 1 and block_j == 1:
        return '.'
    
    # For other blocks, recursively check the cell in the smaller carpet
    return get_cell(n - 1, i % size, j % size)

# Read input
n = int(input())
size = 3 ** n

# Generate and print the carpet
for i in range(size):
    row = []
    for j in range(size):
        row.append(get_cell(n, i, j))
    print(''.join(row))