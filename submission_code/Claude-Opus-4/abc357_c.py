def generate_carpet(n):
    if n == 0:
        return [['#']]
    
    # Get the previous level carpet
    prev_carpet = generate_carpet(n - 1)
    prev_size = len(prev_carpet)
    
    # Create new carpet of size 3^n x 3^n
    new_size = prev_size * 3
    carpet = [['.' for _ in range(new_size)] for _ in range(new_size)]
    
    # Fill the 8 blocks (excluding center) with the previous level carpet
    for block_row in range(3):
        for block_col in range(3):
            # Skip the center block (1, 1)
            if block_row == 1 and block_col == 1:
                continue
            
            # Copy the previous carpet to this block
            for i in range(prev_size):
                for j in range(prev_size):
                    carpet[block_row * prev_size + i][block_col * prev_size + j] = prev_carpet[i][j]
    
    return carpet

# Read input
n = int(input())

# Generate the carpet
carpet = generate_carpet(n)

# Print the carpet
for row in carpet:
    print(''.join(row))