import sys

# Read the integer X from standard input
X = int(sys.stdin.readline())

# Initialize the sum of integers that are not equal to X
total_sum_not_x = 0

# Iterate through the rows of the 9x9 grid (from 1 to 9)
for i in range(1, 10):
    # Iterate through the columns of the 9x9 grid (from 1 to 9)
    for j in range(1, 10):
        # Calculate the value in the cell at row i, column j
        value = i * j
        
        # Check if the calculated value is not equal to X
        if value != X:
            # If it's not equal to X, add it to the running sum
            total_sum_not_x += value

# Print the final calculated sum to standard output
print(total_sum_not_x)