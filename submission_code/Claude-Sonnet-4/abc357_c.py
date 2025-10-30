# YOUR CODE HERE
def get_carpet_cell(n, i, j):
    if n == 0:
        return '#'
    
    # Size of each block in level-n carpet
    block_size = 3 ** (n - 1)
    
    # Determine which block (0-8) the position (i,j) belongs to
    block_row = i // block_size
    block_col = j // block_size
    
    # If it's the center block (1,1), it's white
    if block_row == 1 and block_col == 1:
        return '.'
    
    # Otherwise, it's a level-(n-1) carpet
    # Find position within the block
    local_i = i % block_size
    local_j = j % block_size
    
    return get_carpet_cell(n - 1, local_i, local_j)

n = int(input())
size = 3 ** n

for i in range(size):
    row = ""
    for j in range(size):
        row += get_carpet_cell(n, i, j)
    print(row)