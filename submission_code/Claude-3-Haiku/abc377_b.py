# Read the input from stdin
grid = [input() for _ in range(8)]

# Initialize the count of empty squares
empty_squares = 0

# Iterate through the grid
for i in range(8):
    for j in range(8):
        # Check if the square is empty
        if grid[i][j] == '.':
            # Assume the piece can be placed on this square
            can_place = True
            
            # Check if the piece can be captured by any existing pieces
            for k in range(8):
                # Check the row
                if grid[i][k] == '#':
                    can_place = False
                    break
                # Check the column
                if grid[k][j] == '#':
                    can_place = False
                    break
            
            # If the piece can be placed, increment the count
            if can_place:
                empty_squares += 1

# Print the answer
print(empty_squares)