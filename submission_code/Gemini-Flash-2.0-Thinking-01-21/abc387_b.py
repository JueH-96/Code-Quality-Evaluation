# YOUR CODE HERE
# Read the input integer X from stdin
X = int(input())

# Initialize the sum of integers that are not X
total_sum_not_X = 0

# Iterate through the rows of the 9x9 multiplication table (from 1 to 9)
for i in range(1, 10):
    # Iterate through the columns of the 9x9 multiplication table (from 1 to 9)
    for j in range(1, 10):
        # Calculate the value in the cell at the i-th row and j-th column
        product = i * j

        # If the value in the cell is not equal to X, add it to the total sum
        if product != X:
            total_sum_not_X += product

# Print the final sum of integers that are not X
print(total_sum_not_X)