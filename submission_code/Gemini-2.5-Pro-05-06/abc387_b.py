# Read the input integer X
X = int(input())

# Initialize the sum of elements that are not X
total_sum_not_X = 0

# Iterate through the 9x9 grid
# Rows are indexed from 1 to 9
for i in range(1, 10):  # i will take values 1, 2, ..., 9
    # Columns are indexed from 1 to 9
    for j in range(1, 10):  # j will take values 1, 2, ..., 9
        # Calculate the value in the cell (i-th row, j-th column)
        cell_value = i * j
        
        # Check if this cell's value is not equal to X
        if cell_value != X:
            # If it's not X, add it to our running sum
            total_sum_not_X += cell_value

# Print the final sum
print(total_sum_not_X)